//
//  ReminderListViewController+Actions.swift
//  Today
//
//  Created by LeoLu on 2022/5/15.
//

import UIKit

extension ReminderListViewController {
    
    @objc func didPressDoneButton(_ sender: ReminderDoneButton) {
        guard let id = sender.id else { return }
        completeReminder(with: id)
    }
}
