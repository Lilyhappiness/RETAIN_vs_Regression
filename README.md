# RETAIN_vs_Regression

Runs logistic regression on the claims data in the folder given by [inputpath] and [subfolders] (optional) and
writes results to [resultspath]. Will create [resultspath] if it does not exist. If [resultspath] does exist,
will prompt user before deleting it.

[labelthru] parameter gives the maximum number of days in the last [labelthru] days that will have their
time-varying variables (tvvs) labeled with their index relative to the last claims day. regression.py will then
iterate on x-1 through x=0 (default, no labels).

E.g. if labelthru=2, will label tvvs on last claims day with "1_" and tvvs in second to last claims day
with "2_" and run regression on these sequences. Then will label tvvs on last claims day with "1_" and
run regression on these sequences. Then will label NO tvvs with day index and run regression.

Assumes the following file naming format in path/[subfolder]:

claims_visits_ed_1_data.pkl     # default train data
claims_visits_ed_1_target.pkl   # default train labels
claims_visits_ed_2_data.pkl     # default dev data
claims_visits_ed_2_target.pkl   # default dev labels
claims_visits_ed_3_data.pkl     # default test data
claims_visits_ed_3_target.pkl   # default test labels
claims_visits_ip_1_data.pkl     # default train data
claims_visits_ip_1_target.pkl   # default train labels
claims_visits_ip_2_data.pkl     # default dev data
claims_visits_ip_2_target.pkl   # default dev labels
claims_visits_ip_3_data.pkl     # default test data
claims_visits_ip_3_target.pkl   # default test labels
dictionary.pkl                  # index: feature string mapping

By default, regression.py will train on datasets labeled with "1" and test on datasets labeled with "3". This may
be overridden by the user with [train] and [test] parameters.

Usage:
inputpath and resultspath are required
-- indicates an optional parameter
--help in the command line for help
python regression.py [inputpath] [resultspath] [--subfolders SUBFOLDER1 .. SUBFOLDERn] [--events ed ip] [--train 1 2 3] [--test 1 2 3] [--labelthru INT>=0]

Example 1:

    file structure:
    C:/Users/lli/Downloads/1Match/
        staticonly/
            claims_visits_ed_1_data.pkl ...
        excludelastday/
            claims_visits_ed_1_data.pkl ...
        chemoradonlylastday/
            claims_visits_ed_1_data.pkl ...

    python regression.py "C:/Users/lli/Downloads/1Match" "C:/Users/lli/Downloads/my_results_1" --subfolders excludelastday chemoradonlylastday

Examples 2 and 3:

    file structure:
    C:/Users/lli/Downloads/1Match/
        claims_visits_ed_1_data.pkl ...

    python regression.py "C:/Users/lli/Downloads/1Match" "C:/Users/lli/Downloads/my_results_2" --events ed

    python regression.py "C:/Users/lli/Downloads/1Match" "C:/Users/lli/Downloads/my_results_3" --labelthru 3 --train 1 --test 1
"""
