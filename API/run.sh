#!/bin/bash
redis-server --daemonize yes
sudo mosquitto -c /etc/mosquitto/conf.d/mosquitto.conf
#Install mongodb: https://hevodata.com/blog/install-mongodb-on-ubuntu/
sudo systemctl start mongodb
# sudo systemctl status mongodb
python server.py
