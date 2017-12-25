##算法介绍
###CNN图像识别
####算法介绍
![](http://upload-images.jianshu.io/upload_images/8920871-4df70ba1699211d5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
使用CNN神经网络对六百多张图片进行学习，判断小车应当直走、左转、还是右转。如左图所示，白线斜率过大，小车距离白线过近，因此小车应该左转，如中间图片所示，小车应该直走，如右图所示，视野内并没有白线，此时默认小车直走。
####CNN神经网络的基本结构
![](http://upload-images.jianshu.io/upload_images/8920871-8a1060796aa069b4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
可以看出最左边的图像是输入层，计算机理解为输入若干个矩阵，接着是卷积层（Convolution Layer），在卷积层后面是池化层(Pooling layer)，卷积层+池化层的组合可以在隐藏层出现很多次，在若干卷积层+池化层后面是全连接层（Fully Connected Layer, 简称FC），最后是输出层。
1. 卷积层
卷积层是CNN神经网络中最重要的一层，我们通过如下的一个例子来理解它的原理。图中的输入是一个二维的3x4的矩阵，而卷积核是一个2x2的矩阵。这里我们假设卷积是一次移动一个像素来卷积的，那么首先我们对输入的左上角2x2局部和卷积核卷积，即各个位置的元素相乘再相加，得到的输出矩阵S的S00S00的元素，值为aw+bx+ey+fzaw+bx+ey+fz。接着我们将输入的局部向右平移一个像素，现在是(b,c,f,g)四个元素构成的矩阵和卷积核来卷积，这样我们得到了输出矩阵S的S01S01的元素，同样的方法，我们可以得到输出矩阵S的S02，S10，S11，S12S02，S10，S11，S12的元素。
![](http://upload-images.jianshu.io/upload_images/8920871-f4df5aaacfbf2428.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
2. 池化层
池化层的作用是对输入张量的各个子矩阵进行压缩。假如是2x2的池化，那么就将子矩阵的每2x2个元素变成一个元素，如果是3x3的池化，那么就将子矩阵的每3x3个元素变成一个元素，这样输入矩阵的维度就变小了。

要想将输入子矩阵的每nxn个元素变成一个元素，那么需要一个池化标准。常见的池化标准有2个，MAX或者是Average。即取对应区域的最大值或者平均值作为池化后的元素值。

下面这个例子采用取最大值的池化方法。同时采用的是2x2的池化。步幅为2。
![](http://upload-images.jianshu.io/upload_images/8920871-723fe51b10311831.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

首先对红色2x2区域进行池化，由于此2x2区域的最大值为6.那么对应的池化输出位置的值为6，由于步幅为2，此时移动到绿色的位置去进行池化，输出的最大值为8.同样的方法，可以得到黄色区域和蓝色区域的输出值。最终，我们的输入4x4的矩阵在池化后变成了2x2的矩阵。进行了压缩。
3. 损失层
dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。
![](http://upload-images.jianshu.io/upload_images/8920871-84e202734475fffe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
dropout最重要的功能就是防止数据出现过拟合。
####算法具体实现
1. CNN结构图
使用keras搭建卷积神经网络
![](http://upload-images.jianshu.io/upload_images/8920871-7510f47e3e607f0f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
2. CNN各层介绍
- 卷积层*2：3*3小核计算，降低复杂度同时不损失精度
- 激活层：Relu，f(x)=max(0,x)，收敛速度快
- 池化层：区域压缩为1/4，降低复杂度并减少特征损失
- 全连接层*2：将分布式特征表示映射到样本标记空间
- Dropout层：Dropout设为0.5，防止过拟合，减少神经元之间相互依赖
- 激活层：softmax，平衡多分类问题
3. 效果分析
![](http://upload-images.jianshu.io/upload_images/8920871-39dc901b83d08761.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
上图是我们各个类别的准确率和召回率。可以看出，除了类别1，也就是左转类的召回率较低以外，其他类的准确率和召回率都较高。
![](http://upload-images.jianshu.io/upload_images/8920871-21e2d32374808fcc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
宏平均（Macro-averaging），是先对每一个类统计指标值，然后在对所有类求算术平均值。
微平均（Micro-averaging），是对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标。
4. 具体代码实现
```
model = Sequential()  

model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),  
                        padding='same',  
                        input_shape=(200,480,1))) # 卷积层
model.add(Activation('relu')) #激活层
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层2  
model.add(Activation('relu')) #激活层  
model.add(MaxPooling2D(pool_size=pool_size)) #池化层
model.add(Dropout(0.25)) #神经元随机失活
model.add(Flatten()) #拉成一维数据
model.add(Dense(128)) #全连接层1
model.add(Activation('relu')) #激活层  
model.add(Dropout(0.5)) #经过交叉验证
model.add(Dense(nb_classes)) #全连接层2  
model.add(Activation('softmax')) #评分函数
  
#编译模型  
model.compile(loss='categorical_crossentropy',  
              optimizer='adadelta',  
              metrics=['accuracy'])  
#训练模型  
model.fit(train, y, batch_size=32, epochs=3,  
          verbose=1)
```
###实时学习
####算法介绍
1. 识别出运动的像素点
![](http://upload-images.jianshu.io/upload_images/8920871-99f9717936f2bd72.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
通过对比相邻的两帧图像之间像素点的移动，标注出移动的像素点。得到效果图如下图所示。
![](http://upload-images.jianshu.io/upload_images/8920871-4ec9499c539d0303.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
代码如下所示：
```
def draw_flow(old, new, step=4):
    flow = cv.calcOpticalFlowFarneback(
        cv.cvtColor(old, cv.COLOR_BGR2GRAY), 
        cv.cvtColor(new, cv.COLOR_BGR2GRAY), 
        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    h, w = new.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1)
    fx, fy = flow[np.int32(y), np.int32(x)].T

    lines = np.int32(np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2))

    for (x1, y1), (x2, y2) in lines:
        if sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) < 15:
            continue
        cv.line(old, (x1, y1), (x2, y2), (0, 128, 0), 2)
        cv.circle(old, (x1, y1), 3, (0, 255, 0), -1)
        # x1 y1是old的运动点坐标，x2y2是new运动点的坐标
    return old
```
2. 画出目标区域
kmeans 算法接受参数 k ；然后将事先输入的n个数据对象划分为 k个聚类以便使得所获得的聚类满足：同一聚类中的对象相似度较高；而不同聚类中的对象相似度较小。聚类相似度是利用各中对象的均值所获得一个“中心对象”（引力中心）来进行计算的。

Kmeans算法是最为经典的基于划分的聚类方法，是十大经典数据挖掘算法之一。Kmeans算法的基本思想是：以空间中k个点为中心进行聚类，对最靠近他们的对象归类。通过迭代的方法，逐次更新各聚类中心的值，直至得到最好的聚类结果。
![](http://upload-images.jianshu.io/upload_images/8920871-8012bcd409797bcb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
我们使用Kmeans聚类分析的算法，将运动的像素点划分为三个类别，分别用矩形框将区域框出。
![](http://upload-images.jianshu.io/upload_images/8920871-86c64b93e10fb08b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
3. 特征提取
我们使用颜色作为人的主要特征，找出上步标注出的三个矩形框中面积最大的一个，进行主颜色的提取。
```
def get_dominant_color(image):  
      
#颜色模式转换，以便输出rgb颜色值  
    image = image.convert('RGBA')  
      
#生成缩略图，减少计算量，减小cpu压力  
    image.thumbnail((200, 200))  
      
    max_score = 0
    dominant_color = 0
      
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):  
        # 跳过纯黑色  
        if a == 0:  
            continue  
          
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]  
         
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)  
         
        y = (y - 16.0) / (235 - 16)  
          
        # 忽略高亮色  
        if y > 0.9:  
            continue  
          
        score = (saturation + 0.1) * count  
          
        if score > max_score:  
            max_score = score  
            dominant_color = (r, g, b)  
      
    return dominant_color
```
4. 实时识别
我们根据颜色特征来识别出实时图像中人的位置。在RGB颜色空间中，以主颜色+-20作为判断的颜色区域，找出符合的像素点。通过erode和dilate来平滑像素点，得到一个区域，然后通过opencv的轮廓寻找功能找到区域轮廓的像素点，用矩形框标出这个区域。
![](http://upload-images.jianshu.io/upload_images/8920871-ec752bde2d3741dd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
mask = cv2.inRange(image, lower, upper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

if len(cnts) > 0:  
#找到面积最大的轮廓  
	c = max(cnts, key = cv2.contourArea)
	x1,y1 = 1000,1000
	x2,y2 = 0,0
    
	for i in range(0,len(c)):
		if c[i][0][0] < x1:
			x1 = c[i][0][0]
		if c[i][0][0] > x2:
			x2 = c[i][0][0]
		if c[i][0][1] < y1:
			y1 = c[i][0][1]
		if c[i][0][1] > y2:
			y2 = c[i][0][1]

cv2.rectangle(image,(x1,y1),(x2,y2),(55,255,155),5)
```
