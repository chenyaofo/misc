kubectl apply -f namespace.yaml
kubectl create configmap nginx-cm --from-file=conf.d/server.conf -n my-nginx
kubectl apply -f pv.yaml
kubectl apply -f pvc.yaml
kubectl apply -f nginx-deployment.yaml
kubectl apply -f svc.yaml