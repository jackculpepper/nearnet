function save_avi(outfilename, infilefmtstr, s_frame, i_frame, f_frame)

flag_display = 0;

mov = avifile(outfilename,'colormap',(0:1/255:1)'*ones(1,3),'fps',15);
%mov = avifile(outfilename,'fps',15);

for j = s_frame:i_frame:f_frame
    frame = imread(sprintf(infilefmtstr,j));

    if (flag_display)
        figure(4);
        imagesc(frame); axis image off;
        drawnow;
    end

    mov = addframe(mov,frame);

    fprintf(' %d',j);
end
mov = close(mov);

