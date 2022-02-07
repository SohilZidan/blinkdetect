* singularity container: library://szidane/default/rt_bene:latest
datasets are on nas/datasets so you need to bind them before 

* change zidan
    ```bash
    singularity instance start --nv \
    --bind /nas/database/rt-bene:/home/zidan/rt-bene,\
    /nas/database/rt-gene:/home/zidan/rt-gene,\
    /nas/database/talkingFace:/home/zidan/blinkdetection/dataset/talkingFace,\
    "/nas/database/RN(Researcher's Night)":/home/zidan/blinkdetection/dataset/RN \
    library://szidane/default/rt_bene:latest rt1_instance
    ```

* currently I dont have an updated version of my repo on bitcucket, but you can clone the one I have on nipg

* this is not necessary if you don't use my blinkdetect package
    ```bash
    export PYTHONPATH=$HOME/blinkdetection
    ```
* rt-bene related scripts are at: home/zidan/blinkdetection/scripts/rtbene
    you can find the whole pipeline example at /home/zidan/blinkdetection/scripts/rtbene/pipeline.sh

* rt-bene clips: I generated them at /home/zidan/rtbene_clips using the below command
* you can read the file /home/zidan/temporal_rtbene_v1/rtbene_raw_HDF5.h5 as it contains Dataframe, with the first index the subject, and the range as the second index, and columns are annotations and filepaths (related to my username)

```bash
python3 ./blinkdetection/scripts/rtbene/build_complete_clips.py \
    --annotations /home/zidan/temporal_rtbene_v1/rtbene_raw_HDF5.h5 \
    --output ./rtbene_clips
```

* you can find several bash scripts at /home/zidan/blinkdetection/examples/rtbene, the names are self-explanatory.