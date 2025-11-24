docker rmi -f colorize/api
docker rmi -f colorize/web
docker build -f ./src/deployment/api/Dockerfile.local -t colorize/api .
docker build -f ./src/deployment/app/Dockerfile.local -t colorize/web .