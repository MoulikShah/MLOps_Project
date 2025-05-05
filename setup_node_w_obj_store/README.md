-First create a node/instance.  
-Associate to the floating IP  
-Open the terminal and ssh to the ip  
ssh -i ~/.ssh/id_rsa_chameleon cc@floating_ip
  
#Create the rclone config  
mkdir -p ~/.config/rclone  
nano ~/.config/rclone/rclone.conf  
  
#Here enter the rclone config (The secret and id is our whatsapp grp lol)  
[chi_tacc]  
type = swift  
user_id = PUT_USER_ID_FROM_CHITACC_IDENTITY_USERS  
application_credential_id = CREDENTIAL_ID  
application_credential_secret = CREDENTIAL_SECRET  
auth = https://chi.tacc.chameleoncloud.org:5000/v3  
region = CHI@TACC  
  
Now we are ready to download docker and rclone, mount to our object store, and open up a docker container with access to our data where you can play around with our object store data  
  
Run:  
bash <(curl -fsSL https://raw.githubusercontent.com/MoulikShah/MLOps_Project/main/setup_node_w_obj_store/setup.sh)  
  
Now enter the Jupyter environment with the url with floating IP and run your experiments  
