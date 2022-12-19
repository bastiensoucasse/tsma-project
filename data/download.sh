#!/bin/zsh

if [[ $(basename $(pwd)) != "data" ]]; then
    cd data

    if [[ $(basename $(pwd)) != "data" ]]; then
        exit
    fi
fi

base_url="https://dept-info.labri.fr/~hanna/ProjetClassif"

typeset -A files

files[melspectro_x_train.pickle]="melspectro_songs_train_new.pickle"
files[melspectro_y_train.pickle]="melspectro_genres_train_new.pickle"
files[melspectro_ids_test.pickle]="melspectro_filenames_test.pickle"
files[melspectro_x_test.pickle]="melspectro_songs_test_new.pickle"

files[vggish_train.pickle]="train_id_genres_vgg.pickle"
files[vggish_test.pickle]="test_id_vgg.pickle"

files[openl3_train.pickle]="train_openl3.pickle"
files[openl3_test.pickle]="test_openl3.pickle"

for file uri in "${(@kv)files}"; do
    if [ -f $file ]; then
        echo "\"$file\" already downloaded."
    else
        echo "Downloading \"$file\"…"
        curl -s -o $file $base_url/$uri
    fi
done
