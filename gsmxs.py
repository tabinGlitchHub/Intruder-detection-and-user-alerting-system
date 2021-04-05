import sys
from GSMTC35.GSMTC35 import GSMTC35

gsm = GSMTC35()

if not gsm.setup(_port="COM3"):
    print("Setup error")
    sys.exit(2)

if not gsm.isAlive():
    print("The GSM module is not responding...")
    sys.exit(2)


def send_sms(contact, text):
    # print("SMS sent: " + str(gsm.sendSMS(contact, text, 0.2)))
    print("intruder!")
