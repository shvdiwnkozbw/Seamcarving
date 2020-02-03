import numpy as np
import cv2
from matplotlib import pyplot as plt


class SeamCarver:
    def __init__(self, filename, out_height, out_width, protect_mask='', object_mask='', demo=False, fast_mode=False):
        # 参数初始化
        self.filename = filename
        self.out_height = out_height
        self.out_width = out_width
        self.demo = demo
        self.fast_mode = fast_mode

        # for object removal to see clearly
        self.number_of_carving = 0

        # 读图片
        self.in_image = cv2.imread(filename, 1).astype(np.float64)
        #self.in_image = cv2.cvtColor(self.in_image, cv2.COLOR_BGR2RGB)
        self.in_image = np.array(self.in_image).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]

        # 初始化输出图片
        self.out_image = np.copy(self.in_image)

        # 模式选择
        # 这部分是不是考虑直接省去？可以直接用相关的函数。
        # 当然，没时间就懒得改动了。
        self.object = (object_mask != '')
        if self.object:
            # 注意这里读入mask是灰度图，下面protect同理
            self.mask = cv2.imread(object_mask, 0).astype(np.float16)
#            self.mask = 255 - self.mask
            self.protect = False
        
        self.protect = (protect_mask != '')
        if self.protect:
            if self.object:
                protect = cv2.imread(protect_mask, 0).astype(np.float16)
                self.mask = self.mask - protect
            else:
                self.mask = cv2.imread(protect_mask, 0).astype(np.uint8)
        # 能量图计算中所用到的核
        self.kernel_x = np.array(
            [[0., 0., 0.], 
            [-1., 0., 1.], 
            [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array(
            [[0., 0., 0.], 
            [0., 0., 1.], 
            [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array(
            [[0., 0., 0.], 
            [1., 0., 0.], 
            [0., -1., 0.]], dtype=np.float64)

        # the constant used to minimize energy
        self.constant = 1e10
        self.penalty = 1e10

        # 开始计算
        self.start()


    def start(self):
        '''
        所有流程的开始函数
        '''
        if self.object:
            self.object_removal()
        else:
            self.seams_carving()
        print('Progress finished!')
        if self.demo:
            self.show_pics()

    def show_pics(self):
        '''
        demo function
        '''
        demo_out = np.array(self.out_image).astype(np.uint16)
        demo_out = cv2.cvtColor(demo_out, cv2.COLOR_BGR2RGB)
        demo_in = np.array(self.in_image).astype(np.uint16)
        demo_in = cv2.cvtColor(demo_in, cv2.COLOR_BGR2RGB)

        '''Ploting'''
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(demo_in)
        plt.title('Original')
        plt.subplot(1,2,2)
        plt.imshow(demo_out)
        plt.title('Result')
        plt.show()

    def seams_carving(self):
        '''
        包含了除object_removal之外的操作
        即所有的尺寸变化操作的集合
        '''
        # calculate number of rows and columns needed to be inserted or removed
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        # remove column
        if delta_col < 0:
            self.seams_removal(delta_col * -1)
        # insert column
        elif delta_col > 0:
            self.seams_insertion(delta_col)

        # remove row
        if delta_row < 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            if self.protect:
                self.mask = self.rotate_mask(self.mask, 1)
            self.seams_removal(delta_row * -1)
            self.out_image = self.rotate_image(self.out_image, 0)
        # insert row
        elif delta_row > 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            if self.protect:
                self.mask = self.rotate_mask(self.mask, 1)
            self.seams_insertion(delta_row)
            self.out_image = self.rotate_image(self.out_image, 0)

    def object_removal(self):
        '''
        物体移除操作函数
        '''
        rotate = False
        object_height, object_width = self.get_object_dimension()
        if object_height < object_width:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.mask = self.rotate_mask(self.mask, 1)
            rotate = True

        if self.fast_mode:
            '''
            加速版本，计算一次energy map寻找多条seam
            '''
            self.num_pixel = 10
            while len(np.where(self.mask[:, :] > 0)[0]) > 0:
                energy_map = self.calc_energy_map()
                energy_map[np.where(self.mask[:, :] > 0)] += -self.constant
                energy_map[np.where(self.mask[:, :] < 0)] += self.constant
                cumulative_map = self.cumulative_map_forward(energy_map)
                seam_records, num_seam = self.find_seam(cumulative_map, self.fast_mode)
                for seam_idx in seam_records:
                    assert np.max(seam_idx) < self.out_image.shape[1]
                    self.delete_seam(seam_idx)
                    self.delete_seam_on_mask(seam_idx)
                    seam_records[seam_records>seam_idx] -= 1
                    if len(np.where(self.mask[:, :] > 0)[0]) == 0:
                        break
        else:
            '''一直循环到mask为1的部分全部消失，也即mask全部消失'''
            while len(np.where(self.mask[:, :] > 0)[0]) > 0:
                print('{} pixels remaining!'.format(len(np.where(self.mask[:, :] > 0)[0])))
                energy_map = self.calc_energy_map()
                '''这里，将mask所在之处的能量设为非常低，为负常数值'''
                energy_map[np.where(self.mask[:, :] > 0)] += -self.constant
                energy_map[np.where(self.mask[:, :] < 0)] += self.constant
                cumulative_map = self.cumulative_map_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                self.delete_seam(seam_idx)
                self.delete_seam_on_mask(seam_idx)

        self.mask = - self.mask
        if not rotate:
            num_pixels = self.in_width - self.out_image.shape[1]
        else:
            num_pixels = self.in_height - self.out_image.shape[1]
        self.seams_insertion(num_pixels)
        if rotate:
            self.out_image = self.rotate_image(self.out_image, 0)
            #self.mask = self.rotate_mask(self.mask, 0)


    def seams_removal(self, num_pixel):
        '''
        seam carving函数所需要使用的
        单独移除seam
        '''
        if self.fast_mode:
            '''
            加速版本，计算一次energy map寻找多条seam
            '''
            if self.protect:
                self.num_pixel = num_pixel
                while True:
                    energy_map = self.calc_energy_map()
                    energy_map[np.where(self.mask > 0)] += self.constant
                    cumulative_map = self.cumulative_map_forward(energy_map)
                    seam_records, num_seam = self.find_seam(cumulative_map, self.fast_mode, True)
                    for seam_idx in seam_records:
                        assert np.max(seam_idx) < self.out_image.shape[1]
                        self.delete_seam(seam_idx)
                        self.delete_seam_on_mask(seam_idx)
                        seam_records[seam_records>seam_idx] -= 1
                    self.num_pixel = self.num_pixel - (num_seam+1)
                    if self.num_pixel == 0:
                        break

            else:
                self.num_pixel = num_pixel
                while True:
                    energy_map = self.calc_energy_map()
                    cumulative_map = self.cumulative_map_forward(energy_map)
                    seam_records, num_seam = self.find_seam(cumulative_map, self.fast_mode)
                    for seam_idx in seam_records:
                        assert np.max(seam_idx) < self.out_image.shape[1]
                        self.delete_seam(seam_idx)
                        seam_records[seam_records>seam_idx] -= 1
                    self.num_pixel = self.num_pixel - (num_seam+1)
                    if self.num_pixel == 0:
                        break
        else:
            if self.protect:
                for dummy in range(num_pixel):
                    energy_map = self.calc_energy_map()
                    energy_map[np.where(self.mask > 0)] += self.constant
                    cumulative_map = self.cumulative_map_forward(energy_map)
                    seam_idx = self.find_seam(cumulative_map)
                    self.delete_seam(seam_idx)
                    self.delete_seam_on_mask(seam_idx)
            else:
                for dummy in range(num_pixel):
                    energy_map = self.calc_energy_map()
                    cumulative_map = self.cumulative_map_forward(energy_map)
                    seam_idx = self.find_seam(cumulative_map)
                    self.delete_seam(seam_idx)


    def seams_insertion(self, num_pixel):
        '''
        在能量最低的位置插入seam扩大图片，为防止仅在一个位置插入多条seam，
        接用先裁剪确定n个seam位置，后在原图插入seam的方法
        '''
        if self.fast_mode:
            '''
            加速版本，计算一次energy map寻找多条seam
            '''
            if self.protect:
                temp_image = np.copy(self.out_image)
                temp_mask = np.copy(self.mask)
                self.num_pixel = num_pixel
                seams = []
                while True:
                    energy_map = self.calc_energy_map()
                    energy_map[np.where(self.mask[:, :] > 0)] += self.constant
                    cumulative_map = self.cumulative_map_backward(energy_map)
                    seam_records, num_seam = self.find_seam(cumulative_map, self.fast_mode, insertion=True)
                    for idx, seam_idx in enumerate(seam_records):
                        assert np.max(seam_idx) < self.out_image.shape[1]
                        self.delete_seam(seam_idx)
                        self.delete_seam_on_mask(seam_idx)
                        seam_records[idx:][seam_records[idx:]>seam_idx] -= 1
                        seams.append(seam_idx)
                    self.num_pixel = self.num_pixel - (num_seam+1)
                    if self.num_pixel == 0:
                        break
                seams = np.array(seams)
                self.out_image = np.copy(temp_image)
                self.mask = np.copy(temp_mask)
                for seam in seams:
                    self.add_seam(seam)
                    self.add_seam_on_mask(seam)
                    seams[seams>=seam] += 2
            else:
                self.num_pixel = num_pixel
                seams = []
                temp_image = np.copy(self.out_image)
                while True:
                    energy_map = self.calc_energy_map()
                    cumulative_map = self.cumulative_map_backward(energy_map)
                    seam_records, num_seam = self.find_seam(cumulative_map, self.fast_mode, insertion=True)
                    for idx, seam_idx in enumerate(seam_records):
                        assert np.max(seam_idx) < self.out_image.shape[1]
                        self.delete_seam(seam_idx)
                        seam_records[idx:][seam_records[idx:]>seam_idx] -= 1
                        seams.append(seam_idx)
                    self.num_pixel = self.num_pixel - (num_seam+1)
                    if self.num_pixel == 0:
                        break
                seams = np.array(seams)
                self.out_image = np.copy(temp_image)
                for seam in seams:
                    self.add_seam(seam)
                    seams[seams>=seam] += 2
        else:
            if self.protect:
                temp_image = np.copy(self.out_image)
                temp_mask = np.copy(self.mask)
                seams_record = []
    
                for dummy in range(num_pixel):
                    energy_map = self.calc_energy_map()
                    energy_map[np.where(self.mask[:, :] > 0)] *= self.constant
                    cumulative_map = self.cumulative_map_backward(energy_map)
                    seam_idx = self.find_seam(cumulative_map)
                    seams_record.append(seam_idx)
                    self.delete_seam(seam_idx)
                    self.delete_seam_on_mask(seam_idx)
    
                self.out_image = np.copy(temp_image)
                self.mask = np.copy(temp_mask)
                n = len(seams_record)
                for dummy in range(n):
                    seam = seams_record.pop(0)
                    self.add_seam(seam)
                    self.add_seam_on_mask(seam)
                    seams_record = self.update_seams(seams_record, seam)
            else:
                temp_image = np.copy(self.out_image)
                seams_record = []
    
                for dummy in range(num_pixel):
                    energy_map = self.calc_energy_map()
                    cumulative_map = self.cumulative_map_backward(energy_map)
                    seam_idx = self.find_seam(cumulative_map)
                    seams_record.append(seam_idx)
                    self.delete_seam(seam_idx)
    
                self.out_image = np.copy(temp_image)
                n = len(seams_record)
                for dummy in range(n):
                    seam = seams_record.pop(0)
                    self.add_seam(seam)
                    seams_record = self.update_seams(seams_record, seam)


    def calc_energy_map(self):
        '''
        计算此时的能量势函数
        '''
        b, g, r = cv2.split(self.out_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy


    def cumulative_map_backward(self, energy_map):
        '''
        后向传播
        用于insertion里用于增加seam的操作
        '''
        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                output[row, col] = \
                    energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
        return output


    def cumulative_map_forward(self, energy_map):
        '''
        前向传播
        用于移除seam的操作
        '''
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                if col == 0:
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_right, e_up)
                elif col == n - 1:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_up)
                else:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
        return output


    def calc_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.out_image)
        output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return output


    def fast_process_pixel(self, row, col, mask, cumulative_map, insertion=False):
        '''
        加速版本中，对像素的处理，先满足最低能量路径，以此类推，可能存在路径上像素点被选过情况
        若该像素点能通过其连通区域继续延长最小能量路径，则继续进行
        若该像素点连通区域内无可选，则回溯
        '''
        mask[row, col] = 1
        prev = col
        thres = 1e3 if insertion else np.inf
        if prev == 0:
            if np.sum(mask[row-1, : 2]==0) == 0:
                return False, row+1, 0
            col = np.argmin(cumulative_map[row-1, :2]+self.penalty*self.constant*mask[row-1, :2])
            if (cumulative_map[row-1, col]-np.min(cumulative_map[row-1, :2])) > thres:
                return False, row+1, 0
            return True, row-1, col
        else:
            if np.sum(mask[row-1, prev-1: prev+2]==0) == 0:
                return False, row+1, 0
            col = np.argmin(cumulative_map[row-1, prev-1: prev+2]+self.penalty*self.constant*mask[row-1, prev-1: prev+2]) + prev - 1
            if (cumulative_map[row-1, col]-np.min(cumulative_map[row-1, prev-1: prev+2])) > thres:
                return False, row+1, 0
            return True, row-1, col


    def find_seam(self, cumulative_map, fast_mode=False, insertion=False):
        '''
        找到能量最小的那个seam
        '''
        if fast_mode:
            '''
            加速版本，对于输入的energy map，动态寻找多条seam，
            直到达到终止条件，或达到一定阈值需要重新计算energy map
            '''
            mask = np.zeros_like(cumulative_map)
            print(mask.shape)
            m, n = cumulative_map.shape
            output = np.zeros((m, self.num_pixel), dtype=np.uint32)
            rank = list(np.argsort(cumulative_map[-1]))
            if insertion:
                rank = rank[:2*self.num_pixel]
            for idx in range(self.num_pixel):
#                print(idx)
                row = m
                prev = np.sum(mask)
                while True:
                    if row == 0:
                        mask[row, col] = 1
                        break
                    if row == m:
                        try:
                            col = rank.pop(0)
                        except:
                            return output[:, : idx].transpose(), idx-1
                        if (cumulative_map[-1, col]-np.min(cumulative_map[-1])) > 1e3:
                            return output[:, :idx].transpose(), idx-1
                        row = row - 1
                        output[row, idx] = col
                        mask[row, col] = 1
                    else:
                        success, row, col = self.fast_process_pixel(row, col, mask, cumulative_map, insertion)
                        if success:
                            output[row, idx] = col
                        if not success:
                            col = output[row, idx] if row < m else 0
                if np.sum(mask) - prev > 10 * m:
                    return output[:, : idx].transpose(), idx-1
            return output.transpose(), idx
        else:
            m, n = cumulative_map.shape
            output = np.zeros((m,), dtype=np.uint32)
            output[-1] = np.argmin(cumulative_map[-1])
            for row in range(m - 2, -1, -1):
                prv_x = output[row + 1]
                if prv_x == 0:
                    output[row] = np.argmin(cumulative_map[row, : 2])
                else:
                    output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
            return output


    def delete_seam(self, seam_idx):
        '''
        删除一个seam
        '''

        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)


    def add_seam(self, seam_idx):
        '''
        根据已有的seam idx增加一条seam
        '''
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n + 1, 3))
        for row in range(m):
            col = seam_idx[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.out_image[row, col: col + 2, ch])
                    output[row, col, ch] = self.out_image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
                else:
                    p = np.average(self.out_image[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = self.out_image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
        self.out_image = np.copy(output)


    def update_seams(self, remaining_seams, current_seam):
        '''
        在扩大的时候使用
        更新，增加current_seam到
        seam的表中
        '''
        output = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            output.append(seam)
        return output


    def rotate_image(self, image, ccw):
        m, n, ch = image.shape
        output = np.zeros((n, m, ch))
        if ccw:
            image_flip = np.fliplr(image)
            for c in range(ch):
                for row in range(m):
                    output[:, row, c] = image_flip[row, :, c]
        else:
            for c in range(ch):
                for row in range(m):
                    output[:, m - 1 - row, c] = image[row, :, c]
        return output


    def rotate_mask(self, mask, ccw):
        m, n = mask.shape
        output = np.zeros((n, m))
        if ccw > 0:
            image_flip = np.fliplr(mask)
            for row in range(m):
                output[:, row] = image_flip[row, : ]
        else:
            for row in range(m):
                output[:, m - 1 - row] = mask[row, : ]
        return output


    def delete_seam_on_mask(self, seam_idx):
        '''
        在mask当中删除一个seam
        '''
        m, n = self.mask.shape
        output = np.zeros((m, n - 1))
        for row in range(m):
            col = seam_idx[row]
            output[row, : ] = np.delete(self.mask[row, : ], [col])
        self.mask = np.copy(output)


    def add_seam_on_mask(self, seam_idx):
        '''
        在mask中加入一个seam
        用于insertion中，有protection的情况
        '''
        m, n = self.mask.shape
        output = np.zeros((m, n + 1))
        for row in range(m):
            col = seam_idx[row]
            if col == 0:
                p = np.average(self.mask[row, col: col + 2])
                output[row, col] = self.mask[row, col]
                output[row, col + 1] = p
                output[row, col + 1: ] = self.mask[row, col: ]
            else:
                p = np.average(self.mask[row, col - 1: col + 1])
                output[row, : col] = self.mask[row, : col]
                output[row, col] = p
                output[row, col + 1: ] = self.mask[row, col: ]
        self.mask = np.copy(output)


    def get_object_dimension(self):
        '''
        获取object removal中
        object的尺寸
        '''
        rows, cols = np.where(self.mask > 0)
        height = np.amax(rows) - np.amin(rows) + 1
        width = np.amax(cols) - np.amin(cols) + 1
        return height, width


    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))



