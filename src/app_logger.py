import logging
#configure the root logger
logger = logging.basicConfig(
level=logging.DEBUG, #SHOW ONLY LOGS THAT ARE EQUAL OR ABOVE THIS LEVEL
format='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s',
handlers=[ #Output each log to app.log file and stdout (terminal) 
logging.FileHandler("app.log"),
logging.StreamHandler()])
# Get the root logger
logger = logging.getLogger() 
