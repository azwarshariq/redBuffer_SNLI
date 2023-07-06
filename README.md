# redBuffer_SNLI
<h2>SNLI_task</h2>
<h3>Extract zip folder (snli_1.0)</h3>

```
docker pull azwarshariq/snli_classifier
```
```
docker run -it --name snli_classifier azwarshariq/snli_classifier:latest 
```
<h3>Gunicorn</h3>
Build a new image from the existing docker file to handle multiple requests to Flask Server 
<br />

```
docker run -p 3000:8000 azwarshariq/snli_classifier:latest
```
