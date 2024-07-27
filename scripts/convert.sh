#!/bin/bash

convert_hair_data_app="/disk3/proj_hair/stylist/build/RelWithDebInfo/bin/convert-hair-data-app"
input_dir=$1
output_dir=$2

hairs=($(find ${input_dir}/ -type f -name "*.data")) # works with subdirectories
# convert to other formats
formats=("abc")
for format in ${formats[@]}
do
    output=${output_dir}/${format}
    rm -rf ${output}
    mkdir -p ${output}
    for hair in ${hairs[@]}
    do
        filename=$(basename -- ${hair})
        stem=${filename%.data}
        ${convert_hair_data_app} --input=${hair} --output=${output}/${stem}.${format} --randomcolor
    done
done