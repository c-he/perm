function getConst(id) {
    switch (id) {
        case "me":
            document.write("<a target='_blank' href='https://c-he.github.io'>Chengan He</a>");
            break;
        case "holly":
            document.write("<a target='_blank' href='https://graphics.cs.yale.edu/people/holly-rushmeier'>Holly Rushmeier</a>");
            break;
        case "julie":
            document.write("<a target='_blank' href='https://graphics.cs.yale.edu/people/julie-dorsey'>Julie Dorsey</a>");
            break;
        case "tkim":
            document.write("<a target='_blank' href='http://www.tkim.graphics/'>Theodore Kim</a>");
            break;
        case "yizho":
            document.write("<a target='_blank' href='https://zhouyisjtu.github.io/'>Yi Zhou</a>");
            break;
        case "jsaito":
            document.write("<a target='_blank' href='https://scholar.google.com/citations?user=NftcSfwAAAAJ&hl=en'>Jun Saito</a>");
            break;
        case "jz":
            document.write("<a target='_blank' href='https://jameszachary.com/'>James Zachary</a>");
            break;
        case "fluan":
            document.write("<a target='_blank' href='https://luanfujun.com/'>Fujun Luan</a>");
            break;
        case "yangtwan":
            document.write("<a target='_blank' href='https://tuanfeng.github.io/'>Tuanfeng Y. Wang</a>");
            break;
        case "xinsun":
            document.write("<a target='_blank' href='https://www.sunxin.name/'>Xin Sun</a>");
            break;
        case "zshu":
            document.write("<a target='_blank' href='https://zhixinshu.github.io/'>Zhixin Shu</a>");
            break;
        case "soren":
            document.write("<a target='_blank' href='https://storage.googleapis.com/pirk.io/index.html'>Sören Pirk</a>");
            break;
        case "alejandro":
            document.write("<a target='_blank' href='https://cemse.kaust.edu.sa/people/person/jorge-alejandro-amador-herrera'>Jorge Alejandro Amador Herrera</a>");
            break;
        case "dmichels":
            document.write("<a target='_blank' href='http://dmichels.de/'>Dominik L. Michels</a>");
            break;
        case "mengzephyr":
            document.write("<a target='_blank' href='https://mengzephyr.com/'>Meng Zhang</a>");
            break;
        case "yfeng":
            document.write("<a target='_blank' href='https://yfeng95.github.io/'>Yao Feng</a>");
            break;
        case "giljoo":
            document.write("<a target='_blank' href='https://sites.google.com/view/gjnam'>Giljoo Nam</a>");
            break;
        case "junxuan":
            document.write("<a target='_blank' href='https://junxuan-li.github.io/'>Junxuan Li</a>");
            break;
        case "tobias":
            document.write("<a target='_blank' href='https://tobias-kirschstein.github.io/'>Tobias Kirschstein</a>");
            break;
        case "artem":
            document.write("<a target='_blank' href='https://seva100.github.io/'>Artem Sevastopolsky</a>");
            break;
        case "shunsuke":
            document.write("<a target='_blank' href='https://shunsukesaito.github.io/'>Shunsuke Saito</a>");
            break;
        case "qytan":
            document.write("<a target='_blank' href='https://qytan.com/'>Qingyang Tan</a>");
            break;
        case "javier":
            document.write("<a target='_blank' href='https://scholar.google.com/citations?user=Wx62iOsAAAAJ&hl=en'>Javier Romero</a>");
            break;
        case "chencao":
            document.write("<a target='_blank' href='https://sites.google.com/site/zjucaochen/home'>Chen Cao</a>");
            break;
        case "junx":
            document.write("<a target='_blank' href='https://junxnui.github.io/'>Jun Xing</a>");
            break;
        case "hongzhi":
            document.write("<a target='_blank' href='http://hongzhiwu.com/'>Hongzhi Wu</a>");
            break;
        case "yiwei":
            document.write("<a target='_blank' href='hhttps://yiweihu.netlify.app/'>Yiwei Hu</a>");
            break;
        case "valentin":
            document.write("<a target='_blank' href='https://valentin.deschaintre.fr/'>Valentin Deschaintre</a>");
            break;
        case "kaizhang":
            document.write("<a target='_blank' href='http://cocoakang.cn/'>Kaizhang Kang</a>");
            break;
        case "kunzhou":
            document.write("<a target='_blank' href='http://kunzhou.net/'>Kun Zhou</a>");
            break;
        case "adobe":
            document.write("<a target='_blank' href='https://research.adobe.com/'>Adobe Research</a>");
            break;
        case "mihoyo":
            document.write("<a target='_blank' href='https://www.mihayo.com/'>miHoYo <span lang='zh-cn'>(米哈游)</span></a>");
            break;
        case "yale-graphics":
            document.write("<a target='_blank' href='https://graphics.cs.yale.edu'>Computer Graphics Group</a>");
            break;
        case "zju":
            document.write("<a target='_blank' href='https://www.zju.edu.cn/english/'>Zhejiang University</a>");
            break;
        case "email":
            document.write("chengan <w>dot</w> he <w>at</w> yale <w>dot</w> edu");
            break;
        default:
            document.write("<font color=\"#FF0000\">UNDEFINED</font>");
    }
}

function getSep(size) {
    switch (size) {
        case "big":
            document.write("&emsp;|&emsp;");
            break;
        case "small":
            document.write("&ensp;|&ensp;");
            break;
        default:
            document.write("UNDEFINED");
    }
}

function sup(text, link) {
    var fontsize = "";
    return `<font size="${fontsize}"><a target='_blank' href="${link}">[${text}]</a></font>`;
}

function paper(link) {
    document.write(sup("paper", link));
}

function poster(link) {
    document.write(sup("poster", link));
}

function video(link) {
    document.write(sup("video", link));
}

function talk(link) {
    document.write(sup("talk", link));
}

function thesis(link) {
    document.write(sup("thesis", link));
}

function code(link) {
    document.write(sup("code", link));
}

function data(link) {
    document.write(sup("data", link));
}

function demo(link) {
    document.write(sup("demo", link));
}

function doc(link) {
    document.write(sup("doc", link));
}

function supp(link) {
    document.write(sup("supp", link));
}

function project(link) {
    document.write(sup("project", link));
}

function bibtex(link) {
    document.write(sup("bibtex", link));
}

function hl(text) {
    document.write(`<font color="#ED2224">${text}</font>`);
}

function toggleNews() {
    var moreNews = document.getElementById("moreNews");
    var moreNewsBtn = document.getElementById("moreNewsBtn");
    var lessNewsBtn = document.getElementById("lessNewsBtn");
    if (moreNewsBtn.style.display === "none") {
        moreNews.style.display = "none";
        moreNewsBtn.style.display = "inline";
        lessNewsBtn.style.display = "none";
    } else {
        moreNews.style.display = "inline";
        moreNewsBtn.style.display = "none";
        lessNewsBtn.style.display = "inline";
    }
}
