import os
from os import getcwd
from PIL import Image
import json
import argparse

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
    
    
def main(args): 

    """ Configure Paths"""   
    basepath = os.path.dirname(args.dir)
    labelpath = os.path.join(args.dir, "label")
    outpath = os.path.join(args.dir, "result")

    wd = getcwd()
    #list_file = open('%s_list.txt'%(wd), 'w')

    label_dict = dict()
    f = open("./label.txt", "r")
    lines = f.readlines()
    label_list = [line.strip() for line in lines]
    for i in label_list:
        label_dict[i.split(" ")[1]] = i.split(" ")[0]

    """ Get input json file list """
    json_name_list = []
    for file in os.listdir(labelpath):
        if file.endswith(".json"):
            json_name_list.append(file)
        

    """ Process """
    for json_name in json_name_list:
        txt_name = json_name.rstrip(".json") + ".txt"
        """ Open input text files """
        txt_path = os.path.join(labelpath, json_name)
        print("Input:" + txt_path)
        txt_file = open(txt_path, "r")
        
        """ Open output text files """
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        txt_outpath = os.path.join(outpath, txt_name)
        print("Output:" + txt_outpath)
        txt_outfile = open(txt_outpath, "a")

        """ Convert the data to YOLO format """ 
        # lines = txt_file.read().split('\r\n')   #for ubuntu, use "\r\n" instead of "\n"
        # lines = txt_file.read().split('\r\n')
        with open(txt_path) as json_file:
            lines = json.load(json_file)
        
        for idx, line in enumerate(lines["shapes"]):
            if ("lineColor" in line):
                break 	#skip reading after find lineColor
            if ("label" in line):
                x1 = float(line['points'][0][0])
                y1 = float(line['points'][0][1])
                x2 = float(line['points'][1][0])
                y2 = float(line['points'][1][1])

                # cls = str(0) if line['label'] == "belt" else str(1)
                cls = label_dict[line['label']] 
                

                #in case when labelling, points are not in the right order
                xmin = min(x1,x2)
                xmax = max(x1,x2)
                ymin = min(y1,y2)
                ymax = max(y1,y2)
                img_path = str(f'%s/{basepath}/original/%s.jpg'%(wd, os.path.splitext(json_name)[0]))

                im=Image.open(img_path)
                w= int(im.size[0])
                h= int(im.size[1])

                print(w, h)
                print(xmin, xmax, ymin, ymax)
                b = (xmin, xmax, ymin, ymax)
                bb = convert((w,h), b)
                print(bb)
                txt_outfile.write(cls + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./6classes/0919_dataset/")
    args = parser.parse_args()

    main(args)