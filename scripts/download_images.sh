while read -r -a columns; do
	instance_id=${columns[0]}
	docker_image=${columns[1]}
	baidu_image=${columns[2]}
	echo ${instance_id}
	#echo ${baidu_image}

	docker pull ${docker_image}
	docker tag ${docker_image} ${baidu_image}
	docker push ${baidu_image}
	docker rmi ${docker_image}
	docker rmi ${baidu_image}

done < "b.txt"
