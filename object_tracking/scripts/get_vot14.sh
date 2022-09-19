#!/bin/bash

save_directory=$1
echo "Set dataset directory to ${save_directory}"

vot14_dataset=(
    "ball" "https://data.votchallenge.net/vot2014/dataset/ball.zip" "https://data.votchallenge.net/sequences/9291477fe4b1066f47c12e593078a5ca5caed289e3f3ef24b38370643ae47baa5fc2fee3f69c809923d00859d36d9798b3d86cce88b15220d5813f23a49bc60e.zip"
    "basketball" "https://data.votchallenge.net/vot2014/dataset/basketball.zip" "https://data.votchallenge.net/sequences/743e130c408b3ca236e98668cd8997909e8142f7e6caf87f1a2b1f546de1dc8977d0b92585c41a500b8116718a8513137fbc054a17c907401983011a4ed4aa6e.zip"
    "bicycle" "https://data.votchallenge.net/vot2014/dataset/bicycle.zip" "https://data.votchallenge.net/sequences/84b35f0d0e6a1e39eea057d2141542688a51b1bc17bce9c6bf582b9ea744d2b3d705bb2238661481f20edc68e303002977968ac12201a7c0a8c66c2e9ce963bd.zip"
    "bolt" "https://data.votchallenge.net/vot2014/dataset/bolt.zip" "https://data.votchallenge.net/sequences/99d2404e08700fa77acfb72a8f98044f64a03a4021cceb720e7500f72dc1cb919253305e3cdabdfe8bd854555461c5e15076aa1e61d77ef58f1d84a5a3272307.zip"
    "car" "https://data.votchallenge.net/vot2014/dataset/car.zip" "https://data.votchallenge.net/sequences/ced11f174a535573a4da2d06c7885e5c23955302d5ea7a910c963bb99643d1b03e2e2cd5cf5b7650e461b835f68e7f0e0c03c271e2bbbc571859fe1f5d54ce06.zip"
    "david" "https://data.votchallenge.net/vot2014/dataset/david.zip" "https://data.votchallenge.net/sequences/0b55211166a515fef8a37153589f82ef9deca547810d7034f362b5f254599e0295a406d24db4975a8e78023d145ca936530d296a3fa83f1e574f62cfc7ffbe52.zip"
    "diving" "https://data.votchallenge.net/vot2014/dataset/diving.zip" "https://data.votchallenge.net/sequences/89e9791307ba507b3bc2d2f7e3b124ae0fc0a0076cc983f51a7b7a99d75b5a4cf6e7741a9eb53012d497ab2d5fc166b0dea517ecd87805afda7295466ef733af.zip"
    "drunk" "https://data.votchallenge.net/vot2014/dataset/drunk.zip" "https://data.votchallenge.net/sequences/2e6a0f63fc014ecb0415910691545029438823b960529c5e415fa038f4709a0e98cdb3c1d5b7067dc4cc1bd62fa8a7c013ef94c036d04fa5f02ef459deac165e.zip"
    "fernando" "https://data.votchallenge.net/vot2014/dataset/fernando.zip" "https://data.votchallenge.net/sequences/f2612d6b9cf3d1c217a4f130c466781351e1d2735e47e119a70e86de3d66051fc308e1ddb52af9e11f6f27427909adc84a6fffb824902473c389ad6ccc151974.zip"
    "fish1" "https://data.votchallenge.net/vot2014/dataset/fish1.zip" "https://data.votchallenge.net/sequences/66d988882679e9cd1eec2c51d11fa5e5f5d959511d7f4d3e0a60aaa23621760811ecf218864a7ca87168b96661a862556efc1554ea8d120d421ae8eedff6d037.zip"
    "fish2" "https://data.votchallenge.net/vot2014/dataset/fish2.zip" "https://data.votchallenge.net/sequences/9470b2f2427e79a75f0ad5ffa5e585aafcf36740c27a63fcdb5665d5858e9ab6ebbce7b5c5bc096c56dbbb04e85d9f8a7b2b25fc31ee220ff0eeb0f21612a54f.zip"
    "gymnastics" "https://data.votchallenge.net/vot2014/dataset/gymnastics.zip" "https://data.votchallenge.net/sequences/8490ac66eed4d46cccf6d203e9401ffc7684cb0f5ec79291cdf7eae3044dd86f9d2899ec4e0b9b579b7f71fba8b5c8238e793f6307548f452158ab98a28d5e82.zip"
    "hand1" "https://data.votchallenge.net/vot2014/dataset/hand1.zip" "https://data.votchallenge.net/sequences/4d3110c148463e3bb79b9710485978ab73823235411da2fdf0963f2ecd1dfd2c7ad1ada26c5cbabdd605e0069919a677627b6d533e12516f4ef0cc8753e0fb66.zip"
    "hand2" "https://data.votchallenge.net/vot2014/dataset/hand2.zip" "https://data.votchallenge.net/sequences/ed4a8769a52847a906d9fd7a696f223d7f38002372a86d7cb708bbb9348b475b39103f58a19cd995a09bffc8f3a0c92d6f5e71858d26471fed78b2593ebeba85.zip"
    "jogging" "https://data.votchallenge.net/vot2014/dataset/jogging.zip" "https://data.votchallenge.net/sequences/9d96cc61181db08acced3e8bc191f98af1c7bd01e42fcbd72fae10c6a682791744b74dd3b3ad42e984d69de48e2c0b25d595185ecfbebdfb87091e71149de0ef.zip"
    "motocross" "https://data.votchallenge.net/vot2014/dataset/motocross.zip" "https://data.votchallenge.net/sequences/43f7a7f9fdd699b943e9d6efec04e39527206c48be48ed20ed1fb5fcb9d118197609e7df48628679669e70a26906e0e4ab8b7986c26fb6f34fd0fa7baf384113.zip"
    "polarbear" "https://data.votchallenge.net/vot2014/dataset/polarbear.zip" "https://data.votchallenge.net/sequences/02a39d21e6beeafa0e15051576e95e04182552a6bc298f65f9f93b0dd6a6782c743c8b91b0cde55818aef5196a29641d1d57cb193193ed8c02452573d55f1d18.zip"
    "skating" "https://data.votchallenge.net/vot2014/dataset/skating.zip" "https://data.votchallenge.net/sequences/85a5e18dd159703174d4719e76477986b8758ec7e36c8469b2d5f252ae4b2a604009bbc70ea9ff8e8687f4f713892ad49aca2d8637080096da6ad670e2f80e3e.zip"
    "sphere" "https://data.votchallenge.net/vot2014/dataset/sphere.zip" "https://data.votchallenge.net/sequences/0114c82bc825425c113e64a326dbc20b453b2167959b7eed6722e7836153ebde0e045c8a455aa364290a47216da1aaeaebf5be24e6468839f5e933edc7f43fc7.zip"
    "sunshade" "https://data.votchallenge.net/vot2014/dataset/sunshade.zip" "https://data.votchallenge.net/sequences/ee8a5493f27c87f3fd228728d17e6f94b97cdd8a6d53dd530fef7f91367683d4e34a91d091cbf1ec3be86f923ba4d24efa620af6f5df366ddaae1431ad3204a3.zip"
    "surfing" "https://data.votchallenge.net/vot2014/dataset/surfing.zip" "https://data.votchallenge.net/sequences/c396ecd98fe5b5dd305c27ecc242a79749affc56e495d880334c25e55b86fbef68cd3db211a3b237fb5c4c90e791a94cc2cf9c2b26475dcf2e2854fc63e30ef4.zip"
    "torus" "https://data.votchallenge.net/vot2014/dataset/torus.zip" "https://data.votchallenge.net/sequences/154dde749c3d9cd9b65d2190642f81648013e1ce78c05ca52cb570eaed310d61f3a5eb980926c1d226a1e76776290aa2b32bc9bc0acf2c69b595b49656ff75f7.zip"
    "trellis" "https://data.votchallenge.net/vot2014/dataset/trellis.zip" "https://data.votchallenge.net/sequences/1427ef9e15edc72f6dcd7c5aef934816e996b71167a72ac1432be96e12e70d2954d9ecc795cc23628fe1c7e434d5b27f0225d7e3dd366fe4d3d3ef1738bd7bd4.zip"
    "tunnel" "https://data.votchallenge.net/vot2014/dataset/tunnel.zip" "https://data.votchallenge.net/sequences/8584ec379aea4a1e6b6ed252ce45038b4dead5050442a966cf0f8dbd23b82750daf2b7045abc1a48e0aca4e9b2465779d70adec4e7cee29208c8e86c71f75aca.zip"
    "woman" "https://data.votchallenge.net/vot2014/dataset/woman.zip" "https://data.votchallenge.net/sequences/1fc978e34637161516698f8634b6d24ab093fa374a5e83929e06da18db4198aff16892a2509b2201c82021f3eef5e2f702ca274d6debb93af3b6bc248ece4ffc.zip"
)

idx=0
num_cols=3
for sample in ${vot14_dataset[@]}; do
    if [ $((idx%num_cols)) -eq 0 ]; then
        sample_name=$sample
        sample_path=${save_directory}/${sample_name}
        mkdir -p ${sample_path}
    fi

    if [ $((idx%num_cols)) -eq 1 ]; then
        wget -O ${sample_path}/annotation.zip ${sample}
        unzip -qq ${sample_path}/annotation.zip -d ${sample_path}/annotation
        rm ${sample_path}/annotation.zip
    fi

    if [ $((idx%num_cols)) -eq 2 ]; then
        wget -O ${sample_path}/images.zip ${sample}
        unzip -qq ${sample_path}/images.zip -d ${sample_path}/images
        rm ${sample_path}/images.zip
    fi
    ((++idx))   # increment idx
done