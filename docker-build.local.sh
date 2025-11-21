docker rmi -f colorize/api
docker rmi -f colorize/web
docker build -f ./src/deployment/api/Dockerfile -t colorize/api .
docker build -f ./src/deployment/app/Dockerfile -t colorize/web .