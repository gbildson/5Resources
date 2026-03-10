jq -Rs '{text: .}' strats.md | \
curl -L \
  -X POST \
  -H "Accept: text/html" \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/markdown \
  -d @- > out3.html
