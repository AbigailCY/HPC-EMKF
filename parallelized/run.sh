# echo " "
# echo "-x5 -y2 -t500"
# ./main_seq -x5 -y2 -t500 -o0 -i1
# ./main_omp1 -x5 -y2 -t500 -o0

# echo " "
# echo "-x16 -y16 -t500"
# ./main_seq -x16 -y16 -t500 -o1
# ./main_omp1 -x16 -y16 -t500 -o1

# echo " "
# echo "-x50 -y50 -t500"
# ./main_seq -x50 -y50 -t500 -o2
# ./main_omp1 -x50 -y50 -t500 -o2

# echo " "
# echo "-x100 -y100 -t200"
# ./main_seq -x100 -y100 -t200 -o3
# ./main_omp1 -x100 -y100 -t200 -o3

# echo " "
# echo "-x150 -y150 -t200"
# ./main_seq -x150 -y150 -t200 -o3
# ./main_omp1 -x200 -y200 -t200 -o3

# echo " "
# echo "-x200 -y200 -t200"
# ./main_seq -x200 -y200 -t200 -o3
# ./main_omp1 -x200 -y200 -t200 -o3

# echo " "
# echo "-x250 -y250 -t10"
# ./main_seq -x250 -y250 -t10 -o3
# ./main_omp1 -x250 -y250 -t10 -o3

# echo " "
# echo "-1000 -1000 -t10"
# ./main_seq -x1000 -y1000 -t10 -o3
# ./main_omp1 -x1000 -y1000 -t10 -o3

SEQ="./main_seq"
PROG="./main_omp1 ./main_omp2"
flag1="-x5 -y2 -t500 -o0"
flag2="-x16 -y16 -t500 -o1"
flag3="-x50 -y50 -t500 -o2"
flag4="-x100 -y100 -t200 -o3"
flag5="-x150 -y150 -t200 -o3"
flag6="-x200 -y200 -t200 -o3"
flag7="-x250 -y250 -t10 -o3"
flag8="-x300 -y300 -t10 -o3"
flag9="-x500 -y500 -t10 -o3"
flag10="-x750 -y750 -t10 -o3"
flag11="-x1000 -y1000 -t10 -o3"
FLAG=("$flag1" "$flag2" "$flag3" "$flag4" "$flag5" "$flag6" "$flag7" "$flag8" "$flag9" "$flag10" "$flag11")
Ntr="2 4 6 8 10 12 14 16 18 20 22 24"
for F in "${FLAG[@]}" ; do
    echo
    echo "${SEQ} ${F}"
    ${SEQ} ${F}
    for P in ${PROG}; do
        for N in ${Ntr}; do
            echo
            echo "${P} ${F} -n${N}" 
            ${P} ${F} -n${N}
        done
    done
done
