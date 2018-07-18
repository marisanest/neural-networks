# Get CIFAR10
# BASEDIR=$(cd $(dirname "$0"); pwd)
# echo "$BASEDIR"
cd $(dirname "$0")
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
