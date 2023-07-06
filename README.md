# redBuffer_SNLI
SNLI_task
Extract zip folder (snli_1.0)
.
.
.
.
docker pull azwarshariq/snli_classifier
docker run -it --name snli_classifier azwarshariq/snli_classifier:latest 

docker run -p 3000:8000 azwarshariq/snli_classifier:latest // assuming mutliple