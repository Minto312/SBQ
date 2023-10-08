mkdir ./images

eval dir_list=("$(ls --quoting-style=shell ./compe/train)")
for i in {0..14}; do
    mkdir ./images/$i
    eval cp ./compe/{train,val}/${dir_list[$i]}/*.jpg ./images/$i/
done