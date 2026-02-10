#!/bin/bash

# Scale up the fashionpedia-api deployment
echo "Current fashionpedia-api pods:"
kubectl get pods -l app=fashionpedia-api

echo -e "\nScaling to 5 replicas..."
kubectl scale deployment fashionpedia-api --replicas=5

echo -e "\nWaiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=fashionpedia-api --timeout=300s

echo -e "\nNew fashionpedia-api pods:"
kubectl get pods -l app=fashionpedia-api

echo -e "\nService endpoints:"
kubectl get endpoints fashionpedia-api

echo -e "\nAll set! The ClusterIP service will automatically load balance across all 5 pods."