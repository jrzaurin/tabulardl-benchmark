# sort imports
isort --quiet .
# Black code style
black .
# flake8 standards
flake8 . --max-complexity=10 --max-line-length=127 --ignore=E203,E266,E501,E722,F401,F403,F405,W503,C901
# # mypy
# mypy pytorch_widedeep --ignore-missing-imports --no-strict-optional