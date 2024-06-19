def MIMIC_Findings(file_path):
    findings = ""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_extracting = False
        for line in lines:
            if line.strip().startswith('FINDINGS:') or line.strip().startswith('IMPRESSION:'):
                start_extracting = True
                findings += line.strip().replace("FINDINGS:", "The chest x-ray findings reveal ").replace("IMPRESSION:", "The impression suggests ") + " "
                continue
            if start_extracting:
                findings += line.strip() + " "
        if findings == "":
            findings = " ".join([line.strip() for line in lines])
    return findings.strip()