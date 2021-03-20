nn(fashion_df_net,[X], O, [0,1,2,3,4,5,6,7,8,9,10]) :: category(X,O).

%shirt achtig 0 
price(0,10).
%trui achtig 1
price(1,25).
%short 2
price(2,20).
%broek casual 3
price(3,25).
%broek normaal 4
price(4,40).
%skirt + dress 5
price(5,35).
%coat 6
price(6,40).

% tank 7
price(7,5).
% button down 8
price(8,15).
% chinos 9
price(9,45).
% anarok 10
price(10,50).



totalPrice(I1, I2, TP) :- 
    category(I1,C1),
    category(I2,C2),
    price(C1,P1),
    price(C2,P2),
    TP is P1 + P2.
