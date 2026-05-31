#!/bin/bash
# Phase 2 MTP sweep — n_max × p_min at temp=0.6, ub=2048
# Measures effective decode tok/s and acceptance rate from server logs.

BINARY=~/pflash-llama.cpp/build/bin/llama-server
MODEL=~/Qwen3.6-27B-UD-Q4_K_XL.gguf
PORT=8091
N_REQ=6
MAX_TOK=200
OUTFILE=~/phase2_results.tsv

PROMPT="Explain the key architectural differences between transformer and state space model (SSM) neural networks. Cover: computational complexity, how each handles long sequences, memory footprint during inference, and what types of tasks each excels at. Be specific and technical."

printf "n_max\tp_min\teff_tg\taccept_rate\taccepted\tgenerated\tpp_tps\n" > $OUTFILE

run_config() {
    local n_max=$1
    local p_min=$2
    local label="n=${n_max} p=${p_min}"
    echo ""
    echo "=== $label ==="

    pkill -f llama-server 2>/dev/null; sleep 2

    $BINARY -m $MODEL -ngl 99 -np 1 -fa 1 -ctk turbo3 -ctv turbo3 \
        -b 4096 -ub 2048 -c 100000 --no-context-shift \
        --spec-type mtp --spec-draft-n-max $n_max --spec-draft-p-min $p_min \
        --reasoning off --jinja --no-warmup \
        --host 0.0.0.0 --port $PORT \
        > /dev/null 2>/tmp/p2_err.log &

    # Wait for ready
    local ready=0
    for i in $(seq 1 30); do
        if curl -s "http://localhost:${PORT}/health" 2>/dev/null | grep -q "ok"; then
            ready=1; break
        fi
        sleep 2
    done
    if [ $ready -eq 0 ]; then
        echo "  FAILED to start"
        pkill -f llama-server 2>/dev/null
        return
    fi
    echo "  server ready, sending ${N_REQ} requests..."

    for i in $(seq 1 $N_REQ); do
        curl -s -X POST "http://localhost:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"qwen35\",
                 \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
                 \"temperature\":0.6,
                 \"max_tokens\":$MAX_TOK}" > /dev/null
        echo -n "."
    done
    echo ""

    # Parse last N_REQ entries from log
    local eff_tg=$(grep -E "eval time" /tmp/p2_err.log | grep -v "prompt" | \
        grep -oP '[\d.]+(?= tokens per second)' | tail -$N_REQ | \
        awk '{s+=$1;n++} END {if(n>0) printf "%.2f",s/n}')

    local pp_tps=$(grep "prompt eval time" /tmp/p2_err.log | \
        grep -oP '[\d.]+(?= tokens per second)' | tail -$N_REQ | \
        awk '{s+=$1;n++} END {if(n>0) printf "%.1f",s/n}')

    local accepted=$(grep "draft acceptance rate" /tmp/p2_err.log | \
        grep -oP '\d+(?= accepted)' | tail -$N_REQ | \
        awk '{s+=$1;n++} END {if(n>0) printf "%d",s/n}')

    local generated=$(grep "draft acceptance rate" /tmp/p2_err.log | \
        grep -oP '\d+(?= generated)' | tail -$N_REQ | \
        awk '{s+=$1;n++} END {if(n>0) printf "%d",s/n}')

    local accept_rate=$(grep "draft acceptance rate" /tmp/p2_err.log | \
        grep -oP '[\d.]+(?= \()' | tail -$N_REQ | \
        awk '{s+=$1;n++} END {if(n>0) printf "%.1f%%",s*100/n}')

    echo "  eff_tg=${eff_tg} tok/s  accept=${accept_rate}  accepted=${accepted}/${generated}  pp=${pp_tps} tok/s"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$n_max" "$p_min" "$eff_tg" "$accept_rate" "$accepted" "$generated" "$pp_tps" \
        >> $OUTFILE

    pkill -f llama-server 2>/dev/null
    wait 2>/dev/null
    sleep 3
}

# --- n_max sweep at p_min=0.75 ---
echo "### n_max sweep (p_min=0.75, temp=0.6, ub=2048) ###"
for n in 1 2 3 4 5; do
    run_config $n 0.75
done

# --- find best n_max from results ---
BEST_N=$(awk 'NR>1 && $3!="" {print $3, $1}' $OUTFILE | sort -rn | head -1 | awk '{print $2}')
echo ""
echo "Best n_max appears to be: $BEST_N"
echo ""

# --- p_min sweep at best n_max ---
echo "### p_min sweep (n_max=${BEST_N}, temp=0.6, ub=2048) ###"
for p in 0.50 0.75 0.90; do
    run_config $BEST_N $p
done

echo ""
echo "=== Phase 2 complete. Results in $OUTFILE ==="
cat $OUTFILE | column -t
