#!/usr/bin/env bash

## get our specific AWS stuff sorted

. ./env.sh

EXP_FILES=configurations/frostbite_es.json


cd ../../

#DEBUG="-m pdb -c continue"
DEBUG=""

python ${DEBUG} scripts/launch.py \
${EXP_FILES} \
--key_name=multipl.ai \
--aws_access_key_id=${AWS_ACCESS_KEY_ID} \
--aws_secret_access_key=${AWS_SECRET_ACCESS_KEY} \
--s3_bucket=deep-neuroev-test \
--spot_price=.05 \
--region_name=us-west-2 \
--zone=us-west-2a \
--cluster_size=50 \
--master_instance_type=t3.medium \
--worker_instance_type=c5.large \
--security_group=deep-neuroev-test
#--security_group=sg-029de02e4593cd230




# NB:!!! apparently groupId must be specified by name and not ID when not using VPC
# (wtf seriously??)
#--security_group=sg-029de02e4593cd230


# --spot_master
# --archive_excludes
# --yes