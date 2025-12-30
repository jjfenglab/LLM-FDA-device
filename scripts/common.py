import json
import numpy as np
from scipy import stats
import scipy.stats as st

def load_jsonl(device_data_file):
    """Load JSONL file"""
    devices = []
    with open(device_data_file, 'r') as f:
        for line in f:
            device = json.loads(line.strip())
            devices.append(device)
    return devices

def wald_confidence_interval_cts(sample, confidence=0.95):
    """Calculate Wald confidence interval for cts"""
    sample_mean = sample.mean()
    sem = np.sqrt(np.var(sample)/sample.size)
    confidence_interval = st.norm.interval(confidence, loc=sample_mean, scale=sem)

    return sample_mean, confidence_interval[0], confidence_interval[1]


def wald_confidence_interval(successes, total, confidence=0.95):
    """Calculate Wald confidence interval for a proportion"""
    if total == 0:
        return 0, 0, 0

    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z * np.sqrt(p * (1 - p) / total)

    lower = max(0, p - margin_of_error)
    upper = min(1, p + margin_of_error)

    return p, lower, upper

def format_ci_string(estimate, lower, upper, numerator=None, denominator=None, sig_figs: int = 2):
    """Format confidence interval as string with requested significant digits"""
    def round_to_sig_figs(x):
        if x == 0:
            return 0
        return round(x, -int(np.floor(np.log10(abs(x)))) + (sig_figs - 1))

    est_rounded = round_to_sig_figs(estimate)
    lower_rounded = round_to_sig_figs(lower)
    upper_rounded = round_to_sig_figs(upper)

    # return f"{est_rounded}={numer_int}/{denom_int} ({lower_rounded}, {upper_rounded})"
    return f"{est_rounded} ({lower_rounded}, {upper_rounded})"