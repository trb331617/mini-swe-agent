"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import os
import json
import logging
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded, Submitted
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class ClaudeCode:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.instance_id = ""
        self.problem_statement = ""
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.workspace_path = "/app"

    def init(self, instance_id):
        self.instance_id = instance_id

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost},
            self.extra_template_vars,
            kwargs,
        )

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.problem_statement = task
        self.messages = []
        # self.add_messages(
        #     self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
        #     self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        # )

        try:
            action = {"command": "rm -f /app/install_claude_code.sh"}
            output = self.env.execute(action)
            output["extra"] = {"actions": [action]}
            self.add_messages(output)

            output = self.env.copy_file_to_container("/workspace/claudecode_system_prompt.txt", '/root/system_prompt.txt')
            output["extra"] = {"actions": [{"command": "docker cp system_prompt.txt"}]}
            self.add_messages(output)

            # step3: prepare problem statement
            
            user_prompt = f"""
{self.problem_statement}

Can you help me implement the necessary changes to this repository so that the issue can be resolved?

---------
# INSTRUCTIONS
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
2. Create a script to reproduce the error and execute it using the BashTool, to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well

Your thinking should be thorough and so it's fine if it's very long.

You should use tools as much as possible, ideally more than 100 times. You should also implement your own tests first before attempting the problem.

I will export your changes and apply suitable test patches to verify if your fix is correct when you finish this task. This means you MUST NOT modify the testing logic or any of the tests in any way!
--------

**Workspace Path**: The repository is located at `{self.workspace_path}`. All file operations should be performed relative to this directory.
"""
            path = "/workspace/problem_statement_claudecode/"
            if not os.path.exists(path):
                os.makedirs(path)
            file_path = os.path.join(path, self.instance_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(user_prompt)
                
            output = self.env.copy_file_to_container(file_path, '/root/')
            output["extra"] = {"actions": [{"command": f"docker cp {file_path}"}]}
            self.add_messages(output)

            # step4: opencode run
            self.step()
        except InterruptAgentFlow as e:
            self.add_messages(*e.messages)
        except Exception as e:
            self.handle_uncaught_exception(e)
            raise
        finally:
            self.save(self.config.output_path)

        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        self.query()
        return self.execute_actions()

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        message = {} 
        # message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        # self.add_messages(message)
        return message

    def execute_actions(self) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        action = {"command": f"IS_SANDBOX=1 /root/.local/bin/claude --dangerously-skip-permissions --system-prompt \"$(cat /root/system_prompt.txt)\" -p \"$(cat /root/{self.instance_id})\""}
        output = self.env.execute(action, timeout=7000)
        output["extra"] = {"actions": [action]}
        self.add_messages(output)

        action = {"command": "git diff"}
        output = self.env.execute(action)
        output["extra"] = {"actions": [action], "exit_status": "Submitted", "submission": output["output"]}
        return self.add_messages(output)
        #return self.add_messages(*self.model.format_observation_messages({}, [output], self.get_template_vars()))

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data

