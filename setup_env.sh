#!/bin/bash

. ve/bin/activate

. multiplai/scripts/env.sh

eval `ssh-agent`
ssh-add ../multiplai.pem
bash -i   # otherwise sadness


