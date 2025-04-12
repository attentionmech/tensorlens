#!/bin/bash
cd webapp
npm install
npm run build
rm -rf ../tensorlens/web/static/*
cp -r dist/* ../tensorlens/web/static/
