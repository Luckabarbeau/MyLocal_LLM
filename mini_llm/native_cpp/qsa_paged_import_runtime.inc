// Bounded prepared-history transport and diagnostics; no model computation.
extern "C" {
int ml_qsa_layer_import_chunk(ml_handle id,uint64_t position,ml_handle raw,
    ml_handle keys,ml_handle values,ml_handle pooled) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_layers,id);
        if (!w->paged || w->invalid || position!=w->length)
            fail(ML_INVALID,"paged QSA chunk requires valid state and exact position");
        auto r=lookup(uploads,raw);
        const uint64_t rows=r->descriptor.ggml_shape[1];
        if (!rows || rows>4096 || rows>w->capacity-position)
            fail(ML_CAPACITY,"paged QSA chunk exceeds 4096 rows or logical capacity");
        std::shared_ptr<UploadedTensor> u[4]={r,lookup(uploads,keys),lookup(uploads,values),nullptr};
        const uint64_t widths[]={128,512,512,128};
        for (int j=0;j<3;++j) layer_shape(u[j],w->engine,widths[j],rows);
        const uint64_t pools=(position+rows)/4-position/4;
        if (pools) { u[3]=lookup(uploads,pooled); layer_shape(u[3],w->engine,128,pools); }
        else if (pooled) fail(ML_INVALID,"paged QSA chunk without completed blocks requires zero pool handle");
        DeviceGuard guard(w->engine->device); StreamCompletion drain{w->engine->stream};
        for (auto x:u) if (x) x->wait();
        cuda_check(cudaMemsetAsync(w->status,0,4,w->engine->stream));
        for (int j=0;j<4;++j) if (u[j]) cuda_check(ml_qsa_layer_finite(w->engine->stream,
            static_cast<const float*>(u[j]->device),(j==3?pools:rows)*widths[j],w->status));
        w->check_status();
        w->invalid=true; w->output_valid=false;
        auto p=w->paged;
        const uint64_t end=position+rows;
        for (uint64_t page=position/p->page_size;page<=(end-1)/p->page_size;++page) p->reserve(page);
        for (uint64_t offset=0;offset<rows;) {
            const uint64_t count=std::min(rows-offset,p->page_size-(position+offset)%p->page_size);
            p->pack(static_cast<const float*>(u[1]->device)+offset*512,
                static_cast<const float*>(u[2]->device)+offset*512,count,w->status);
            w->check_status();
            p->commit(position+offset,count,static_cast<const float*>(u[0]->device)+offset*128);
            offset+=count;
        }
        if (!position) w->copy(w->raw,static_cast<const float*>(u[0]->device),128);
        const uint64_t begin=rows>4?end-4:position;
        for (uint64_t t=begin;t<end;++t)
            w->copy(w->raw+128+(t%4)*128,static_cast<const float*>(u[0]->device)+(t-position)*128,128);
        if (pools) w->copy(w->pooled+(position/4)*128,static_cast<const float*>(u[3]->device),pools*128);
        cuda_check(cudaStreamSynchronize(w->engine->stream));
        p->stats.length=end; p->stats.populated_bytes=end*p->row_bytes;
        w->length=end; w->invalid=false;
    });
}
int ml_qsa_layer_paged_range_for_test(ml_handle id,int which,uint64_t start,uint64_t rows,
    void* destination,uint64_t count) noexcept {
    return boundary([&] {
        auto w=lookup(qsa_layers,id);
        if (!w->paged || w->invalid || which<1 || which>4 || !destination)
            fail(ML_INVALID,"paged QSA range diagnostic invalid");
        const uint64_t length=which==4?w->length/4:w->length;
        const uint64_t width=(which==1 || which==4)?128:512;
        if (start>length || rows>length-start || count<rows*width)
            fail(ML_INVALID,"paged QSA range diagnostic bounds");
        DeviceGuard guard(w->engine->device);
        if (which==4) {
            if (rows) cuda_check(cudaMemcpy(destination,w->pooled+start*128,rows*512,cudaMemcpyDeviceToHost));
        } else w->paged->read(start+rows,which,static_cast<float*>(destination),start);
    });
}
}
