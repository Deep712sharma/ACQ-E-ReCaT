#!/bin/bash
wget -O mainldm/models/ldm/celeba256/celeba-256.zip https://ommer-lab.com/files/latent-diffusion/celeba.zip
wget -O mainldm/models/ldm/lsun_churches256/lsun_churches-256.zip https://ommer-lab.com/files/latent-diffusion/lsun_churches.zip
wget -O mainldm/models/ldm/lsun_beds256/lsun_beds-256.zip https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip



cd mainldm/models/ldm/celeba256
unzip -o celeba-256.zip

cd mainldm/models/ldm/lsun_churches256
unzip -o lsun_churches-256.zip

cd mainldm/models/ldm/lsun_beds256
unzip -o lsun_beds-256.zip

cd ../..