#sudo du -d 1 -h

input_dir="val"
output_dir="val2"
quality=100
delete_temp=100

# Loop through all jpeg files in the input directory
for img in "$input_dir"/*.jpg; do
    i=$((i+1))

    # Get the filename without the directory path
    filename=$(basename "$img")
    
    # Convert and save the image with the new quality
    convert "$img" -quality $quality "$output_dir/$filename"
    if [ "$i" -eq "$delete_temp" ]; then
        sudo rm -rf /tmp/*
        i=0
    fi
done