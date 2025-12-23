ENV_NAME=conan
source ~/anaconda3/etc/profile.d/conda.sh

OPT=`getopt -o cr -l create,reset -- "$@"`
if [ $? != 0 ] ; then
    exit 1
fi
eval set -- "$OPT"

while true
do
    case $1 in
        -c | --create)
            # -create
            # download raganato framework
            read -p "Download raganato framework? [y/N] "
            if [[ $REPLY =~ ^[Yy]$ ]]
            then
            wget -P corpus/ http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
            unzip -d corpus/ corpus/WSD_Evaluation_Framework.zip
            rm corpus/WSD_Evaluation_Framework.zip
            fi

            if [ -d ./corpus ]; then
                # create conda
                conda create -n ${ENV_NAME} python==3.9
                conda activate ${ENV_NAME}
                
                # pip install
                pip install --upgrade pip
                conda install pytorch==1.9.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
                pip install -r bin/requirements.txt
                
                # download modules
                python -c "import nltk; nltk.download('wordnet')"
                
                # conda install
                conda install openjdk
                javac corpus/WSD_Evaluation_Framework/Evaluation_Datasets/Scorer.java

            else
                echo "Not Exist ./corpus"
            fi
            echo "conda activate ${ENV_NAME}"
            shift
            ;;
        -r | --reset)
            # -reset
            conda remove -n ${ENV_NAME} --all
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!" 1>&2
            exit 1
            ;;
    esac
done
