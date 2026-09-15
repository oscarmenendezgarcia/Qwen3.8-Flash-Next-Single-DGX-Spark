# sparkDash on this box

[sparkDash](https://github.com/MiaAI-Lab/sparkDash) is the dashboard upstream
measures with; its decode and prefill benchmarks are what
`docs/checkpoint-quantization-and-spanish-2026-09-14.md` section 6 reports, so
numbers taken here sit beside upstream's published ones without conversion.

These two files live here rather than in the sparkDash clone because that clone
is someone else's repository: a `git clean` or a re-clone would take them with
it. Copy them in after cloning.

    git clone https://github.com/MiaAI-Lab/sparkDash.git ~/Projects/spark-dash
    cd ~/Projects/spark-dash
    cp .env.example .env && sed -i 's/^LLM_PORT=8888/LLM_PORT=8890/' .env
    cp <this-dir>/docker-compose.override.yml .
    docker compose build
    sudo cp <this-dir>/sparkdash.service /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now sparkdash

Two adjustments, both in the override:

- **`restart: "no"`** — the shipped compose uses `restart: always`. Every server
  on this host is owned by systemd; two owners of one lifecycle is how a
  container comes back at the wrong moment.
- **`LLM_PORT=8890`** — the compose hardcodes 8888, which searxng holds here.

It binds `127.0.0.1:5555` only; a non-loopback bind fails closed without
`SPARKDASH_TOKEN`, because the API does not authenticate direct clients and
exposes benchmarks and power controls. From another machine:

    ssh -N -L 5555:127.0.0.1:5555 oscar@<host>

Register this box once, from the UI or:

    curl -X POST http://127.0.0.1:5555/api/sparks -H 'Content-Type: application/json' \
      -d '{"id":"gb10","name":"<hostname>","isLocal":true,"llmPort":8890}'
