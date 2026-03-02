param(
  [string]$Image = "power-forecast-api:0.3"
)

New-Item -ItemType Directory -Force -Path "state" | Out-Null

docker run --rm -p 127.0.0.1:8000:8000 `
  -v "${PWD}\state:/app/state" `
  -e STATE_DIR=/app/state `
  $Image