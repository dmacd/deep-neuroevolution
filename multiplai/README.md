# multipl.ai specific readme notes


        python3 -m venv ve
               
## local testing

Can follow instructions in the root readme except:

- Running redis doenst work unless redis service is stopped and the
 script is run as root, otherwise the socket doesnt get created                

- socket perms need to be 777 (modified default redis config files
)               

*** KILL REDIS BETWEEN RUNS OR YOU WILL END UP WITH STALE DATA
 CONSUMED BY WORKERS ***

Useful incantations:

```
sudo su
service redis-server stop
killall redis-server
. scripts/local_run_redis.sh
```

- Redis state not reset by the default script?

This seems like it should work but doesnt...use the reset incantations above
 to flush redis when its got stale stuff in it
```
redis-cli KEYS '*'
redis-cli FLUSHALL
redis-cli KEYS '*'
```

Kill zombie worker and master processes
`pkill -f es_distributed`


## local training visualization

Default logdir is `logs/`

Start tensorboard: 
`tensorboard --logdir logs/`


#### TODO
There is probably a way to run redis server without root that creates sockets
 my user can read and write. https://serverfault.com/questions/711566/redis
 -server-does-not-create-socket-file may help
               
## distributed 

       
The uber repo is based on https://github.com/openai/evolution-strategies-starter. To get started with running distributed experiments we need to set up our AMIs 

From AMI finder for us-west-2

    us-west-2	bionic	18.04 LTS	arm64	hvm:ebs-ssd	20190617	ami-0a321f0fdd7eef430	hvm   (amazon architecture?)
    us-west-2	bionic	18.04 LTS	amd64	hvm:ebs-ssd	20190627.1	ami-07b4f3c02c7f83d59	hvm   (x86_64 arch)
    us-west-2	bionic	18.04 LTS	arm64	hvm:ebs-ssd	20190627.1	ami-0c579621aaac8bade	hvm   (amazon architecture)
    us-west-2	bionic	18.04 LTS	amd64	hvm:instance-store	20190627.1	ami-0a9ec5b23fb9ba2e7	hvm


Create a script in multiplai/scripts/env.sh that exports `AWS_SECRET_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

Edit `scripts/packer.json`:

- update region
- select base AMI

Install packer: `sudo apt install packer`


Run packer

     cd scripts && packer build packer.json




## launch


Tweak multiplai/scripts/test_launch.sh and launch

- create new access credentials if desired
- create a new key pair if desired
- set up a security group with
    - ssh access
    - allowing all traffic between members of the group 
      (TODO: tighten this up once we know what actual traffic is used. redis perhaps?)

- ensure instance types are compatible with EBS optimization settings



## NEXT STEP:

x fix gym deps in launcher
x pip freeze > requirements.txt
x debug why redis keys arent found
  v maybe config/security issue?
  v or maybe code-rot...
  - long start delay (unclear why)
  - OOMing w/ less than 4gb memory
x visualize results and make sure everything is cool
x verify that frostbite is, in fact, converging
x get a visual indication of aws spend

- then scale up cluster and let in run for a few hours
- run ES-modified algo....pass as param to launcher?
- understand the VINE tool to leverage it


## TODO
x parameterize algo in launch script

x fix ssh-add thingy
x sync command downloads ve which it shouldnt
- resize leaves cluster / jobs / parameters in a bad state? maybe this isnt 
supported

x can i run on other instance types with more than 1 core?
    -> should totally work
    
- pull in spot price calc so i can tell what/where the best deal is
    - check price variance across regions too 

- nicer sync & vis script

## ec2ctl

Setup env and ssh. From repo root:

    source multiplai/scripts/env.sh

    eval `ssh-agent`
    ssh-add ../multiplai.pem
    
    cd scripts/
        
    ./ec2ctl jobs
    
    ./ec2ctl tail <pattern>


## visualization


Sync & visualize

    # rm -rf test-sync
    mkdir test-sync
    ./ec2ctl sync frostbite test-sync

    cd test-sync/frostbite_20190722-125741/ubuntu/deep-neuroevolution/
    python -m scripts.viz 'FrostbiteNoFrameskip-v4' snapshot_iter00000_rew0.h5

TODO: 
- get env id from experiment
- understand what data is available
- how can i see metrics from the workers??


## troubleshooting

"An error occurred (Unsupported) when calling the RunInstances operation: EBS-optimized instances are not supported for your requested configuration. Please check the documentation for supported configurations."

AAARRGH: "EbsOptimized=True" is NOT required for EBS storage and is incompatible with some instance types, specifically, t2s. fuck. lame.  


Regarging 

    ./ec2ctl sync PATTERN DEST

DEST must exist 

### todos


LATER: try out the aws graviton processors OR t3a.nano which is the cheapest!!



## notes on the codebase


- `es_modified` algorithm differs from `es` in that it has support for
 `bc_vectors`
- `ALGO=es` is hardcoded in `launch.py`



  
