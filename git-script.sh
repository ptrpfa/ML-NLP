#! /bin/bash

commitname = $1
git add .
git commit -m '$commitname'
git push origin master
