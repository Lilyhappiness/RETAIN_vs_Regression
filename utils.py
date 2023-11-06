"""
Lily Li
Last modified: 2019 12

Some helper functions for regression.py and make_data.py. Provides tests for whether a feature
is timevarying, or chemo/radiation, or imaging.

WORKS ONLY ON THE STRING REPRESENTATION OF FEATURES

"""


def is_chemorad(feature):
    """ Returns whether this feature is both timevarying AND chemo or radiation (vs static chemo/rad).
    Args:
        feature (str) : feature as string
    Returns:
        bool : True if feature is timevarying chemo/radiation, False otherwise
    """
    # rule out static variable(s) starting with 'RADIATION'
    if feature == 'RADIATION':
        return True
    # capture all agents and chemo regimens
    if feature.startswith('AGENT_CLASS') or feature.startswith('CHEMOREG'):
        return True
    return False


def is_timevarying(feature):
    """ Returns whether this feature is a timevarying feature.
    Args:
        feature (str) : feature as string
    Returns:
        bool : True if feature is timevarying, False otherwise
    """
    # no static name collisions
    timevarying_safe = [
        'AGENT_CLASS', 'ANTIBIOTIC', 'ANTIEMETIC', 'ANTIFUNGAL', 'ANTIVIRAL',
        'ANYCLAIM', 'ANYIP', 'BLOODTRANS', 'CCS', 'CHEMOREG', 'CSF',
        'DRUG_MAINCLASS', 'HCC', 'IP_ICU', 'MENTALSERVICE', 'NARCOTIC', 
        'ONCOLOGYSERVICE', 'PCSERVICE', 'SPECIALTYSERVICE', 'URGENTCARE']
    for tv in timevarying_safe:
        if feature.startswith(tv):
            return True
    # static name collisions (e.g. RADITIATION_DXYEAR), rule out static
    timevarying_collision = ['FRAILTY', 'RADIATION', 'SNF', 'SURGERY']
    for tv in timevarying_collision:
        if tv == feature:
            return True
    return False


def is_imaging(feature):
    """ Returns whether this feature is an imaging feature such as CT scan.
    Args:
        feature (str) : feature as string
    Returns:
        bool : True if feature is imaging, False otherwise
    """
    imaging = {
        'CCS_177', 'CCS_178', 'CCS_179', 'CCS_180', 'CCS_181', 
        'CCS_183', 'CCS_184', 'CCS_185', 'CCS_186', 'CCS_187', 
        'CCS_188', 'CCS_189', 'CCS_190', 'CCS_191', 'CCS_192', 
        'CCS_193', 'CCS_194', 'CCS_195', 'CCS_196', 'CCS_197', 'CCS_198'
    }
    return feature in imaging
