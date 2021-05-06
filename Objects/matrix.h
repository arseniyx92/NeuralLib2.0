//#pragma once
//
//#include<vector>
//#include<cassert>
//
//template<typename T>
//class matrix{
//    size_t row_len, col_len;
//    std::vector<T> data;
//
//
//public:
//
//    matrix(){
//        row_len = col_len = 0;
//    }
//    matrix(size_t row_len, size_t col_len){
//        this->row_len = row_len;
//        this->col_len = col_len;
//
//        data.resize(row_len * col_len);
//    }
//    matrix(const std::vector<std::vector<T>>& data){
//        row_len = data.size();
//        col_len = data.empty() ? 0 : data.back().size();
//
//        this->data.resize(row_len * col_len);
//
//        for(size_t row = 0; row < data.size(); row++){
//            for(size_t col = 0; col < data[row].size(); col++){
//                this->data[row * col_len + col] = data[row][col];
//            }
//        }
//    }
//
//    size_t get_row_len() const {
//        return row_len;
//    }
//    size_t get_col_len() const{
//        return col_len;
//    }
//
//
//    T* operator [](size_t row){
//        return data.data() + row * col_len;
//    }
//    const T* operator [](size_t row) const {
//        return data.data() + row * col_len;
//    }
//
//    template<typename func_t>
//    matrix aggregate(func_t&& agg_func, matrix& Rhs) const {
//        assert(row_len == Rhs.row_len && col_len == Rhs.col_len);
//
//        matrix result(row_len, col_len);
//
//        size_t n = row_len * col_len;
//        for(size_t i = 0; i < n; i++){
//            result.data[i] = agg_func(data[i], Rhs.data[i]);
//        }
//        return result;
//    }
//    template<typename func_t>
//    matrix aggregate(func_t&& agg_func) const {
//        matrix result(row_len, col_len);
//
//        size_t n = row_len * col_len;
//        for(size_t i = 0; i < n; i++){
//            result.data[i] = agg_func(data[i]);
//        }
//        return result;
//    }
//
//    matrix operator + (const matrix& add) const {
//        return aggregate([](const T& a, const T& b){
//            return a + b;
//        }, add);
//    }
//    matrix& operator += (const matrix& add) {
//        return *this = *this + add;
//    }
//
//    matrix operator - (const matrix& add) const {
//        return aggregate([](const T& a, const T& b){
//            return a - b;
//        }, add);
//    }
//    matrix& operator -= (const matrix& add) {
//        return *this = *this - add;
//    }
//
//
//    matrix operator * (const matrix& mult) const{
//        assert(col_len == mult.row_len);
//
//        matrix result(row_len, mult.col_len);
//
//        for (int i = 0; i < row_len; i++)
//            for (int j = 0; j < mult.col_len; j++)
//                for (int k = 0; k < col_len; k++)
//                    result[i][j] += (*this)[i][k] * mult[k][j];
//        return result;
//    }
//
//
//};