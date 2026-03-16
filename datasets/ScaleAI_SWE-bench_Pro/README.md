---
dataset_info:
  features:
  - name: repo
    dtype: string
  - name: instance_id
    dtype: string
  - name: base_commit
    dtype: string
  - name: patch
    dtype: string
  - name: test_patch
    dtype: string
  - name: problem_statement
    dtype: string
  - name: requirements
    dtype: string
  - name: interface
    dtype: string
  - name: repo_language
    dtype: string
  - name: fail_to_pass
    dtype: string
  - name: pass_to_pass
    dtype: string
  - name: issue_specificity
    dtype: string
  - name: issue_categories
    dtype: string
  - name: before_repo_set_cmd
    dtype: string
  - name: selected_test_files_to_run
    dtype: string
  - name: dockerhub_tag
    dtype: string
  splits:
  - name: test
    num_bytes: 23667808
    num_examples: 731
  download_size: 7816820
  dataset_size: 23667808
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---


## Dataset Summary

SWE-Bench Pro is a challenging, enterprise-level dataset for testing agent ability on long-horizon software engineering tasks.

Paper: https://static.scale.com/uploads/654197dc94d34f66c0f5184e/SWEAP_Eval_Scale%20(9).pdf

See the related evaluation Github: https://github.com/scaleapi/SWE-bench_Pro-os

## Dataset Structure

We follow SWE-Bench Verified (https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) in terms of dataset structure, with several extra fields.

## Data Fields

repo (string): Repository identifier - one of 11 repository classes

instance_id (string): Unique identifier for each instance (65-120 characters)

base_commit (string): Git commit hash of the base version (40 characters)

patch (string): The golden code patch/diff (1.44k - 180k characters)

test_patch (string): Test cases related to the patch (325 - 322k characters)

problem_statement (string): Description of the issue being addressed (419 - 8.04k characters)

requirements (string): Project requirements or dependencies (124 - 6.7k characters, may be null)

interface (string): API or interface specifications (1 - 12.2k characters, may be null)

repo_language (string): Programming language of the repository - one of 4 language classes

fail_to_pass (string): Test cases that should pass after patch application (10 - 155k characters)

pass_to_pass (string): Test cases that should continue passing (2 - 532k characters)

issue_specificity (string): Specificity of the issue (12-77 characters)

issue_categories (string): Categories or tags for the issue type

before_repo_set_cmd (string): Repo set command for testing

selected_test_files_to_run (string): Files selected for testing
