%[sample_set,f_data, f_deri]= thermalblock(2,500);
%save('16_500_test')
[sample_set,f_data, f_deri]= thermalblock(4,500)
save('36_500_test')
[sample_set,f_data, f_deri]= thermalblock(4,10000)
save('36_10000')
name1='test/36_10000_test_'
for i=1:10
    ii=num2str(i);
    name = strcat(name1,ii)
    [sample_set,f_data, f_deri]= thermalblock(4,10000);
    save(name)
    %disp([name])
end
% 
% 
% [sample_set,f_data, f_deri]= thermalblock(3,40000)
% save('36_40000')
% name1='test/36_40000_test_'
% for i=1:10
%     ii=num2str(i);
%     name = strcat(name1,ii)
%     [sample_set,f_data, f_deri]= thermalblock(3,40000);
%     save(name)
%     %disp([name])
% end
