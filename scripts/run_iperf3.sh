ip=$1
role=$2

if [[ $role == "send" ]];
then
    iperf3 -c ${ip} -t 8 -P 30 -p 9992
else
    iperf3 -s -p 9992 
fi
