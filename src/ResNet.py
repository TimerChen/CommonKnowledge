import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections
from datetime import datetime
import time
import math
from tensorflow.contrib.layers.python.layers import utils
 
#定义Block类
class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
    """
    定义一个Block类，有三个属性：
    'scope'： 命名空间
    'unit_fn'：单元函数
    'args'：参数
    """
#定义相关函数
#定义降采样方法
def subsample(inputs,factor,scope=None):
    """
    :param inputs: 输入数据
    :param factor: 采样因子,stride步长
    :param scope: 作用域
    :return:
    """
    if factor==1:
        return inputs
    else:
        return slim.max_pool2d(inputs,kernel_size=[1,1],stride=factor,scope=scope)
 
#定义卷积操作方法
def conv2d_same(inputs,num_outputs,kernel_size,stride,scope=None):
    if stride==1:
        return slim.conv2d(inputs,num_outputs,kernel_size,stride=1,padding='SAME',scope=scope)
    else:
        #修整Inputs,对inputs进行补零操作
        padding_total=kernel_size-1
        padding_beg=padding_total//2
        padding_end=padding_total-padding_beg
        inputs=tf.pad(inputs,[[0,0],[padding_beg,padding_end],[padding_beg,padding_end],[0,0]])
        return slim.conv2d(inputs,num_outputs,kernel_size,stride=stride,padding='VALID',scope=scope)
 
#定义堆叠block方法
@slim.add_arg_scope
def stack_block_dense(net,blocks,outputs_collections=None):
    """
    net:input
    blocks:Block的class列表
    outputs_collections:收集各个end_points的collections
    """
    for block in blocks:
        #双层循环，遍历blocks，遍历res unit堆叠
        with tf.variable_scope(block.scope,'block',[net]) as sc:
            #用两个tf.variable_scope将残差学习单元命名为block1/unit_1的形式
            for i ,unit in enumerate(block.args):
                with tf.variable_scope('unit_%d'%(i+1),values=[net]):
                    #利用第二层循环拿到block中的args,将其展开为depth,depth_bottleneck,strdie
                    unit_depth,unit_depth_bottleneck,unit_stride=unit
                    #使用残差学习单元的生成函数unit_fn，顺序的创建并连接所有的残差学习单元
                    net=block.unit_fn(net,
                                      depth=unit_depth,
                                      depth_bottleneck=unit_depth_bottleneck,
                                      stride=unit_stride)
                    #使用utils.collect_named_outputs将输出net添加到collection中
                    net=utils.collect_named_outputs(outputs_collections,sc.name,net)
    return net
 
#定义resnet参数
def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.97,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    #创建resnet通用的arg_scope
    batch_norm_params={
        'is_training':is_training,
        'decay':batch_norm_decay,
        'epsilon':batch_norm_epsilon,
        'scale':batch_norm_scale,
        'updates_collections':tf.GraphKeys.UPDATE_OPS }
    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params ):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],padding='SAME') as arg_sc:
                return arg_sc
 
#定义瓶颈函数（核心方法）
@slim.add_arg_scope
def bottleneck(inputs,depth,depth_bottleneck,stride,outputs_collections=None,scope=None):
    with tf.variable_scope(scope,'bottleneck_v2',[inputs]) as sc:
        #获取输入的最后一个维度，输出通道数
        depth_in=utils.last_dimension(inputs.get_shape(),min_rank=4)
        #对输入进行batch_borm，接着用relu进行预激活
        preact=slim.batch_norm(inputs,activation_fn=tf.nn.relu,scope='preact')
        if depth==depth_in:
            #如果残差单元输入通道和输出通道数一样，就对inputs进行降采样
            shortcut=subsample(inputs,stride,'shortcut')
        else:
            #如果残差单元输入通道与输出通道数不一样，就使用stride步长的1*1卷积改变其通道数，是的输入通道和输出通道数一样
            shortcut=slim.conv2d(preact,depth,[1,1],stride=stride,normalizer_fn=None,activation_fn=None,scope='shortcut')
        #定义残差
        #第一步：1*1，stride=1，输出通道数为depth_bottleneck的卷积
        #第二步：3*3，stride=stride，输出通道数为depth_bottleneck的卷积
        #第三步：1*1，stride=1，输出通道数为depth的卷积
        residual=slim.conv2d(preact,depth_bottleneck,[1,1],stride=1,scope='conv1')
        residual=slim.conv2d(residual,depth_bottleneck,[3,3],stride=stride,scope='conv2')
        residual=slim.conv2d(residual,depth,[1,1],stride=1,scope='conv3')
        output=shortcut+residual
 
        #将结果添加到outputs_collections
        return utils.collect_named_outputs(outputs_collections,sc.name,output)
 
#定义生成resnet_v2的主函数
def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope,'resnet_v2',[inputs],reuse=reuse) as sc:
        end_point_collections=sc.original_name_scope+'_end_points'
        #用slim.arg_scope将slim.conv2d bottleneck stack_blocks_dense 3个函数的参数outputs_collections设置为end_point_collections
        with slim.arg_scope([slim.conv2d,bottleneck,stack_block_dense],outputs_collections=end_point_collections):
            net=inputs
 
            if include_root_block:
                #根据include_root_block标记，创建resnet最前面一层的卷积神经网络
                with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
                    net=conv2d_same(net,64,7,stride=2,scope='conv1')
                net=slim.max_pool2d(net,[3,3],stride=2,scope='pool1')
            #利用stack_blocks_dense将残差学习模块完成
            net=stack_block_dense(net,blocks)
            net=slim.batch_norm(net,activation_fn=tf.nn.relu,scope='postnorm')
 
            if global_pool:
                #根据标记添加平均池化层
                net=tf.reduce_mean(net,[1,2],name='pool5',keep_dims=True)
 
            if num_classes is not None:
                #根据是否有分类数，添加一个输出通道为num_classes的1*1卷积
                net=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope='logits')
 
            #utils.convert_collection_to_dict将collection转化为dict
            end_points=utils.convert_collection_to_dict(end_point_collections)
 
            if num_classes is not  None:
                #添加一个softmax输出层
                end_points['prediction']=slim.softmax(net,scope='prediction')
 
            return net,end_points
 
#定义resnet_v2_50的生成方法
def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'
                 ):
    #设计50层的resnet
    #四个blocks的units数量为3、4、6、3，总层数为（3+4+6+3）*3+2=50
    #钱3个blocks包含步长为2的层，总尺寸244/(4*2*2*2)=7 输出通道变为2048
    blocks=[
        Block('block1',bottleneck,[(256,64,1)]*2+[(256,64,5)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1) ]* 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope)
 
#定义resnet_v2_101的生成方法
def resnet_v2_101(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'
                 ):
    #设计50层的resnet
    #四个blocks的units数量为3、4、23、3，总层数为（3+4+23+3）*3+2=101
    #钱3个blocks包含步长为2的层，总尺寸244/(4*2*2*2)=7 输出通道变为2048
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 5)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope)
 
#定义测试运行时间函数
def time_tensorflow_run(session, target, info_string):
 
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches+num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time()-start_time
 
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %(datetime.now(), i-num_steps_burn_in, duration))
                total_duration += duration
                total_duration_squared += duration*duration
 
    mn = total_duration/num_batches
    vr = total_duration_squared/num_batches-mn*mn
    sd = math.sqrt(vr)
 
    print('%s: %s across %d steps, %.3f +/- %3.3f sec/batch' %(datetime.now(), info_string, num_batches, mn, sd))
if __name__=='__main__':
    batch_size = 32
    height, width = 224, 224
    inputs = tf.random_uniform((batch_size, height, width, 3))
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net, end_points = resnet_v2_50(inputs, 1000)
 
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    num_batches = 100
    time_tensorflow_run(sess, net, 'Forward')

