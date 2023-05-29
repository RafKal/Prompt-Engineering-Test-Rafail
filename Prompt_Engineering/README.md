# Steps to log in

Log in in Gateway with student_thowl as username

Type `ssh th-stable-diffusion@172.16.160.101`

Enter Password

Type `ls` to list all folders

Navigate to `inpainting_pipeline_JanLippemeierTH` with cd `inpainting_pipeline_JanLippemeierTH`

Type ls and use cd to navigate into ai-gen...

Type `docker images` to list all images

type `docker ps -a` to list all containers 

When inside Prompt_engineering type `nvidia-docker build -t th_prompt_engineering:<TAG> .`

Run Container with `nvidia-docker run -it th_prompt_engineering:<TAG>` (wenn ihr sehen wollt, was passiert)

Run Container with `nvidia-docker run -dit th_prompt_engineering:<TAG>` (wenn ihr sehen wollt, was passiert)

Get container Id with `docker ps -a`

In einen Container wechseln: `nvidia-docker exec -it <ContainerID> bash`

Wenn außerhalb vom Container `docker cp <ContainerID>:/usr/src/<DEST> ./Current_Output`

Bilder auf Hostsystem kopieren: `scp DGX:/ EuerPC/asdasd`

DOcker Images löschen: `docker rmi <ImageID>`

Docker COntainer löschen: `docker rm <ContainerID>`

