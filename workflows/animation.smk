# Workflow creating animations

DATES = [
    "2024-07-01",
    "2024-08-01",
    "2024-09-01",
    "2024-10-01",
    "2024-11-01",
    "2024-12-01",
    "2025-01-01",
    "2025-02-01",
    "2025-03-01",
]

rule all:
    # Final desired output: the animated GIF
    input:
        "/tmp/gif/animation.gif"


rule infer:
    """
    Run covvfit for each date. Here we expect it to produce figure.png in /tmp/frames/{date}.
    """
    output:
        "/tmp/frames/{date}/figure.png"
    shell:
        """
        covvfit infer \
            --input /tmp/data.csv \
            --output /tmp/frames/{wildcards.date} \
            --config data/config-demo.yaml \
            --overwrite-output \
            --time-spacing 2 \
            --date-min 2024-06-01 \
            --date-max {wildcards.date} \
            --horizon-date 2025-04-28
        """


rule copy_png:
    """
    Copy the PNG from /tmp/frames/{date}/figure.png
    into /tmp/gif/{date}-figure.png, so we don't overwrite the same filename.
    """
    input:
        "/tmp/frames/{date}/figure.png"
    output:
        "/tmp/gif/{date}-figure.png"
    shell:
        """
        mkdir -p /tmp/gif
        cp {input} {output}
        """


rule animation:
    """
    Combine all renamed PNGs in /tmp/gif/ into an animation.gif.
    """
    input:
        expand("/tmp/gif/{date}-figure.png", date=DATES)
    output:
        "/tmp/gif/animation.gif"
    shell:
        """
        convert \
            -delay 50 \
            -density 500 \
            -layers Optimize \
            -dither None \
            -loop 0 \
            {input} \
            {output}
        """
