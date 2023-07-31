set -ex

docker run --rm -it $(docker build -q . -f Dockerfile.cpu) pytest -n auto
