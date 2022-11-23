
echo "-x5 -y2 -t500\n"
./main_seq -x5 -y2 -t500 -o0
./main4 -x5 -y2 -t500 -o0
./main1 -x5 -y2 -t500 -o0

echo "-x16 -y16 -t500\n"
./main_seq -x16 -y16 -t500 -o1
./main4 -x16 -y16 -t500 -o1
./main1 -x16 -y16 -t500 -o1

echo "-x50 -y50 -t500\n"
./main_seq -x50 -y50 -t500 -o2
./main4 -x50 -y50 -t500 -o2
./main1 -x50 -y50 -t500 -o2

echo "-x100 -y100 -t500\n"
./main_seq -x100 -y100 -t500 -o3
./main4 -x100 -y100 -t500 -o3
./main1 -x100 -y100 -t500 -o3

echo "-x150 -y150 -t200\n"
./main_seq -x150 -y150 -t200 -o3
./main4 -x200 -y200 -t200 -o3
./main1 -x200 -y200 -t200 -o3

echo "-x200 -y200 -t200\n"
./main_seq -x200 -y200 -t200 -o3
./main4 -x200 -y200 -t200 -o3
./main1 -x200 -y200 -t200 -o3