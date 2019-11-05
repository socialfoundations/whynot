pylint --rcfile=.pylintrc whynot -f parseable -r n 
pycodestyle whynot --max-line-length=120 --exclude=world3_app.py
pydocstyle whynot 
