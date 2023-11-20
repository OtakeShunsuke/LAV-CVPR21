set -e

docker build -t my-python-app . 2> build.log
echo "---------------------------------------------"
docker run -dit -v ./BotCL:/work --gpus all --shm-size=16g --name botcl my-python-app