def check():
    import covvfit._cli.infer as infer

    if not hasattr(infer, "infer"):
        raise ValueError("The tool does not work.")

    print("[Status: OK] The tool has been installed properly.")
